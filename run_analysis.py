#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ctypes
import csv
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Force non-interactive background rendering to absolute seal Matplotlib memory leaks
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "8" 
torch.set_num_threads(8)

def free_memory():
    """Forces Python and the OS to reclaim memory completely."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

# ================================
# Configuration
# ================================
@dataclass
class Config:
    model_names = ["Qwen/Qwen3-14B"]
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-raw-v1"
    dataset_mirror: Optional[str] = "https://hf-mirror.com"
    model_mirror_base: Optional[str] = "https://hf-mirror.com"
    max_length = 128
    batch_size = 8
    num_batches = 128
    embedding_sample_size = 10000 
    cov_dtype = torch.float32
    eps = 1e-6
    output_dir = Path("results")

cfg = Config()
cfg.output_dir.mkdir(exist_ok=True)

# ================================
# Helper functions
# ================================
def layernorm_no_params(x: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)

def maybe_sample_rows(x: torch.Tensor, max_rows: Optional[int], seed: int = 42) -> torch.Tensor:
    if not max_rows or x.shape[0] <= max_rows:
        return x
    gen = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:max_rows]
    return x[idx]

def covariance_from_rows(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    return (x.T @ x) / (x.shape[0] - 1)

class CovAccumulator:
    def __init__(self, dim: int, dtype: torch.dtype, device: torch.device = torch.device('cpu')):
        self.dim = dim
        self.dtype = dtype
        self.count = 0
        self.sum = torch.zeros(dim, dtype=dtype, device=device)
        self.sum_xtx = torch.zeros(dim, dim, dtype=dtype, device=device)

    def update(self, x: torch.Tensor) -> None:
        x = x.to(self.dtype)
        self.sum += x.sum(dim=0)
        self.sum_xtx += x.T @ x
        self.count += x.shape[0]

    def covariance(self) -> torch.Tensor:
        if self.count < 2:
            raise ValueError("Not enough samples for covariance")
        mean = self.sum / self.count
        cov = (self.sum_xtx - self.count * torch.outer(mean, mean)) / (self.count - 1)
        return cov

def eigvals_sorted(cov: torch.Tensor, eps: float) -> torch.Tensor:
    eigvals = torch.linalg.eigvalsh(cov.double())
    eigvals = torch.clamp(eigvals, min=eps)
    return torch.flip(eigvals, dims=[0])

def _piecewise_log2_x(x, linear_max: int = 16) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = x.copy()
    mask = x > linear_max
    if np.any(mask):
        step = linear_max - 1
        y[mask] = linear_max + step * np.log2(x[mask] / linear_max)
    return y

def _set_piecewise_log2_xaxis(ax, max_x: int, linear_max: int = 16) -> None:
    ticks = [1]
    if linear_max <= max_x:
        ticks.append(linear_max)
    value = linear_max * 2
    while value <= max_x:
        ticks.append(value)
        value *= 2
    if ticks[-1] != max_x:
        ticks.append(max_x)
    ticks = [t for t in sorted(set(ticks)) if 1 <= t <= max_x]

    tick_positions = _piecewise_log2_x(ticks, linear_max=linear_max)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.set_xlim(
        _piecewise_log2_x([1], linear_max=linear_max)[0],
        _piecewise_log2_x([max_x], linear_max=linear_max)[0],
    )

def plot_loglog_eigs(eigvals: torch.Tensor, out_path: Path, title: str) -> None:
    n = eigvals.shape[0]
    xs = np.arange(1, n + 1)
    ys = eigvals.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(_piecewise_log2_x(xs), ys, linewidth=1.0)
    ax.set_yscale("log")
    _set_piecewise_log2_xaxis(ax, n)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    ax.clear()
    plt.close(fig)

def fit_powerlaw_loglog(eigvals: torch.Tensor) -> tuple[float, float, float]:
    """Fits a power-law y = c * x^k by linear regression on log-log axes.
    Returns (slope_k, intercept_logc, r2).
    """
    ys = eigvals.detach().cpu().numpy().astype(float)
    xs = np.arange(1, ys.shape[0] + 1, dtype=float)
    # Filter positive values for log transform
    mask = ys > 0
    if mask.sum() < 3:
        return float('nan'), float('nan'), float('nan')
    xs = xs[mask]
    ys = ys[mask]
    lx = np.log(xs)
    ly = np.log(ys)
    A = np.column_stack([lx, np.ones_like(lx)])
    coeffs, _, _, _ = np.linalg.lstsq(A, ly, rcond=None)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    fitted = A @ coeffs
    ss_res = float(np.sum((ly - fitted) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return slope, intercept, r2

def compute_svd_metrics(weight: torch.Tensor, device: torch.device) -> tuple[float, float]:
    """Computes both 99% Rank and Entropy from a single SVD pass to save immense memory and time."""
    w_gpu = weight.to(device, dtype=torch.float32)
    s = torch.linalg.svdvals(w_gpu)
    
    # 99% Energy
    energy = torch.cumsum(s * s, dim=0)
    total = energy[-1]
    k_99 = int(torch.searchsorted(energy, 0.99 * total).item() + 1)
    
    # Entropy
    p = s[s > 1e-10] / s[s > 1e-10].sum()
    erank = torch.exp(-torch.sum(p * torch.log(p))) if p.numel() > 0 else torch.tensor(0.0)
    k_ent = int(round(erank.item()))
    
    denom = min(weight.shape)
    
    # Explicit cleanup of huge math buffers
    del w_gpu, s, energy, p, erank
    return k_99 / denom, k_ent / denom

def build_batches(tokenizer, texts, batch_size: int, max_length: int, num_batches: int):
    batches = []
    needed = batch_size * num_batches
    texts = texts[:needed]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        if len(batch_texts) < batch_size:
            break
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        batches.append(enc)
    return batches

# ================================
# Model analysis functions
# ================================
def _get_blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise AttributeError("Unsupported model block layout")

def _get_hidden_size(config) -> int:
    return getattr(config, "n_embd", None) or getattr(config, "hidden_size", None)

def _get_attn_weights(block, hidden_size: int):
    attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
    if attn is None:
        raise AttributeError("Block has no attention module")

    if hasattr(attn, "c_attn") and hasattr(attn, "c_proj"):
        w_qkv = attn.c_attn.weight.detach()
        w_q = w_qkv[:, :hidden_size]
        w_k = w_qkv[:, hidden_size : 2 * hidden_size]
        w_v = w_qkv[:, 2 * hidden_size :]
        w_o = attn.c_proj.weight.detach()
        return w_q, w_k, w_v, w_o

    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    o_proj = getattr(attn, "o_proj", None)
    if all([q_proj, k_proj, v_proj, o_proj]):
        return (
            q_proj.weight.detach(),
            k_proj.weight.detach(),
            v_proj.weight.detach(),
            o_proj.weight.detach(),
        )
    raise AttributeError("Unsupported attention layout")

def analyze_embedding_stats(model, model_name: str, cfg: Config, model_dir: Path, target_device: torch.device) -> None:
    wte_full = model.get_input_embeddings().weight.detach()
    wte_full = layernorm_no_params(wte_full, cfg.eps)
    wte = maybe_sample_rows(wte_full, cfg.embedding_sample_size)
    
    cov_emb = covariance_from_rows(wte)
    eig_emb = eigvals_sorted(cov_emb, cfg.eps)
    plot_loglog_eigs(eig_emb, model_dir / "embedding_eigs.png", f"{model_name} embedding eigs")
    emb_logmean = float(torch.log(eig_emb).mean().item())
    
    del wte, cov_emb, eig_emb

    with open(model_dir / "embedding_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["log_eig_mean"])
        writer.writerow([emb_logmean])

    wte_f = wte_full.to(target_device, dtype=torch.float32)
    norms = torch.linalg.norm(wte_f, dim=1, keepdim=True).clamp(min=cfg.eps)
    wte_unit = wte_f / norms
    
    del wte_f, norms

    vocab_size = wte_unit.shape[0]
    sample_size = min(4096, vocab_size)
    
    if vocab_size < 2:
        nn_mean_cos = float("nan")
        mean_pairwise_cos = float("nan")
    else:
        gen = torch.Generator(device=target_device).manual_seed(42)
        sample_idx = torch.randperm(vocab_size, generator=gen, device=target_device)[:sample_size]
        sample = wte_unit[sample_idx]
        
        sims = sample @ wte_unit.T
        sims[torch.arange(sample_size), sample_idx] = -1.0
        nn_mean_cos = float(sims.max(dim=1).values.mean().item())
        
        # Explicit cleanup of the massive 2.5GB sims matrix immediately
        del sims, sample
        
        sum_vec = wte_unit.double().sum(dim=0)
        sum_sq = float((sum_vec @ sum_vec).item())
        mean_pairwise_cos = (sum_sq - vocab_size) / (vocab_size * (vocab_size - 1))

    del wte_unit, wte_full

    with open(model_dir / "embedding_similarity_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_size", "nn_mean_cosine", "mean_pairwise_cosine"])
        writer.writerow([sample_size, nn_mean_cos, mean_pairwise_cos])
    
    free_memory()

def analyze_effective_ranks_and_entropy(blocks, hidden_size: int, model_name: str, model_dir: Path, device: torch.device) -> None:
    ranks_99 = {"q": [], "k": [], "v": [], "o": []}
    ranks_ent = {"q": [], "k": [], "v": [], "o": []}
    dims = {"q": "", "k": "", "v": "", "o": ""}

    for i, block in enumerate(tqdm(blocks, desc=f"SVD Metrics ({model_name})")):
        w_q, w_k, w_v, w_o = _get_attn_weights(block, hidden_size)

        if i == 0:
            dims["q"] = f"{w_q.shape[0]}x{w_q.shape[1]}"
            dims["k"] = f"{w_k.shape[0]}x{w_k.shape[1]}"
            dims["v"] = f"{w_v.shape[0]}x{w_v.shape[1]}"
            dims["o"] = f"{w_o.shape[0]}x{w_o.shape[1]}"

        rq99, rqE = compute_svd_metrics(w_q, device)
        rk99, rkE = compute_svd_metrics(w_k, device)
        rv99, rvE = compute_svd_metrics(w_v, device)
        ro99, roE = compute_svd_metrics(w_o, device)

        ranks_99["q"].append(rq99); ranks_ent["q"].append(rqE)
        ranks_99["k"].append(rk99); ranks_ent["k"].append(rkE)
        ranks_99["v"].append(rv99); ranks_ent["v"].append(rvE)
        ranks_99["o"].append(ro99); ranks_ent["o"].append(roE)
        
        # Critical Drop
        del w_q, w_k, w_v, w_o
        free_memory()

    # Save 99% Ranks
    with open(model_dir / "effective_ranks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "q", "k", "v", "o"])
        for i in range(len(ranks_99["q"])):
            writer.writerow([i, ranks_99["q"][i], ranks_99["k"][i], ranks_99["v"][i], ranks_99["o"][i]])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    layers = list(range(len(ranks_99["q"])))
    ax.plot(layers, ranks_99["q"], label=f"Q ({dims['q']})")
    ax.plot(layers, ranks_99["k"], label=f"K ({dims['k']})")
    ax.plot(layers, ranks_99["v"], label=f"V ({dims['v']})")
    ax.plot(layers, ranks_99["o"], label=f"Out ({dims['o']})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized effective rank (99%)")
    ax.set_title(f"{model_name} effective rank vs depth")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout()
    fig.savefig(model_dir / "effective_rank_vs_depth.png", bbox_inches='tight')
    plt.close(fig)

    # Save Entropy Ranks
    with open(model_dir / "effective_rank_entropy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "q", "k", "v", "o"])
        for i in range(len(ranks_ent["q"])):
            writer.writerow([i, ranks_ent["q"][i], ranks_ent["k"][i], ranks_ent["v"][i], ranks_ent["o"][i]])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, ranks_ent["q"], label=f"Q ({dims['q']})")
    ax.plot(layers, ranks_ent["k"], label=f"K ({dims['k']})")
    ax.plot(layers, ranks_ent["v"], label=f"V ({dims['v']})")
    ax.plot(layers, ranks_ent["o"], label=f"Out ({dims['o']})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized effective rank entropy")
    ax.set_title(f"{model_name} effective rank entropy vs depth")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    fig.tight_layout()
    fig.savefig(model_dir / "effective_rank_entropy_vs_depth.png", bbox_inches='tight')
    plt.close(fig)
    
def mse_distance_matrix(x: torch.Tensor, y: torch.Tensor = None):
    """
    计算两个点集之间的 MSE 距离矩阵。
    距离定义：mean((x_i - y_j)^2) over dimension.
    x: (N, D), y: (M, D)，若 y=None 则计算内部距离。
    返回 (N, M) 的距离矩阵。
    """
    if y is None:
        y = x
    # 利用 (a-b)^2 = a^2 + b^2 - 2ab
    x2 = torch.mean(x ** 2, dim=1, keepdim=True)   # (N,1)
    y2 = torch.mean(y ** 2, dim=1, keepdim=True)   # (M,1)
    xy = torch.mm(x, y.T) / x.shape[1]             # (N,M)
    dist2 = x2 + y2.T - 2 * xy
    return dist2.clamp(min=0.0)

def estimate_intrinsic_dimension_twonn(vectors: torch.Tensor, subsample: int = 2000) -> float:
    """
    vectors: (N, D) 所有样本点
    subsample: 用于估计的随机子集大小
    返回估计的内在维度 d
    """
    N = vectors.shape[0]
    if N <= 2:
        return float('nan')
    if N > subsample:
        idx = torch.randperm(N, device=vectors.device)[:subsample]
        vectors = vectors[idx]
        N = subsample

    # 计算两两 MSE 距离矩阵
    dist = mse_distance_matrix(vectors)           # (N, N)

    # 排除自身
    dist.fill_diagonal_(float('inf'))
    
    # 最近邻
    r1, idx1 = torch.min(dist, dim=1)
    # 次近邻
    dist.scatter_(1, idx1.unsqueeze(1), float('inf'))
    r2, _ = torch.min(dist, dim=1)

    mu = r2 / r1
    mu = mu[mu > 1.0]

    if mu.numel() < 10:
        return float('nan')

    mu_sorted = torch.sort(mu)[0].cpu().numpy()
    Nmu = len(mu_sorted)
    F = np.arange(1, Nmu + 1) / Nmu               # 经验累积概率
    log_mu = np.log(mu_sorted)
    log_one_minus_F = np.log(1 - F + 1e-12)

    # 线性拟合 (去掉尾部10%，避免噪声)
    trim = max(1, int(0.1 * Nmu))
    X = log_mu[:-trim] if Nmu > trim else log_mu
    y = log_one_minus_F[:-trim]
    if len(X) < 5:
        return float('nan')

    coeffs = np.polyfit(X, y, 1)   # [斜率, 截距]
    d = -coeffs[0]                 # 斜率应为 -d
    return float(d)

def analyze_hidden_states_and_twonn(
    model,
    batches,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    cfg: Config,
    model_name: str,
    model_dir: Path,
) -> None:
    target_device = model.device

    # --- 1. 初始化频谱分析(Spectra)的累加器 ---
    accs = [CovAccumulator(hidden_size, cfg.cov_dtype, device='cpu') for _ in range(num_layers)]
    sum_unit_vecs = [torch.zeros(hidden_size, dtype=cfg.cov_dtype, device='cpu') for _ in range(num_layers)]
    token_counts = [0] * num_layers

    # --- 2. 初始化 Two-NN 的采样缓存 ---
    max_tokens_per_layer = min(cfg.embedding_sample_size, 10000)
    layer_samples = [[] for _ in range(num_layers)]
    twonn_layer_counts = [0] * num_layers

    # --- 3. 统一的数据收集循环（只跑一次前向传播） ---
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Hidden states & Two-NN ({model_name})")):
        batch_dev = {k: v.to(target_device) for k, v in batch.items()}
        attn_mask = batch_dev.get("attention_mask")
        
        # 仅跑一次前向传播
        with torch.inference_mode():
            outputs = model(**batch_dev, output_hidden_states=True, use_cache=False)
            hidden_list = list(outputs.hidden_states)
        
        del outputs
        del batch_dev

        # 逐层提取和处理
        for i in range(1, len(hidden_list)):
            h = hidden_list[i]                     
            idx = i - 1

            h_flat = h.reshape(-1, h.shape[-1])
            if attn_mask is not None:
                token_mask = attn_mask.reshape(-1).bool().to(h_flat.device)
                h_flat = h_flat[token_mask]
                if h_flat.numel() == 0:
                    hidden_list[i] = None
                    continue

            # 【逻辑 A】Two-NN 数据收集 (使用原始未应用额外 LayerNorm 的状态)
            need = max_tokens_per_layer - twonn_layer_counts[idx]
            if need > 0:
                if h_flat.shape[0] > need:
                    idx_perm = torch.randperm(h_flat.shape[0], device=target_device)[:need]
                    h_sample = h_flat[idx_perm].cpu().float()
                else:
                    h_sample = h_flat.cpu().float()
                layer_samples[idx].append(h_sample)
                twonn_layer_counts[idx] += h_sample.shape[0]

            # 【逻辑 B】频谱分析数据收集 (需要应用无参数 LayerNorm)
            x = layernorm_no_params(h_flat, cfg.eps)
            x_cpu = x.cpu().float()

            norm = torch.norm(x_cpu, dim=-1, keepdim=True).clamp(min=cfg.eps)
            x_unit = x_cpu / norm
            
            sum_unit_vecs[idx] += x_unit.sum(dim=0)
            token_counts[idx] += x_unit.shape[0]
            accs[idx].update(x_cpu)
            
            # 释放当前层张量
            del x, x_cpu, norm, x_unit, h
            hidden_list[i] = None

        del hidden_list
        if (batch_idx + 1) % 20 == 0:
            free_memory()

    # --- 4. 后续处理：计算余弦相似度与频谱分析 ---
    print(f"[{model_name}] Processing hidden state spectra plots and csv...")
    avg_cos_sim = []
    for i in range(num_layers):
        n = token_counts[i]
        if n < 2:
            avg_cos_sim.append(float('nan'))
        else:
            sum_vec = sum_unit_vecs[i]
            norm_sq = (sum_vec @ sum_vec).item()
            avg = (norm_sq - n) / (n * (n - 1))
            avg_cos_sim.append(avg)

    with open(model_dir / "avg_cosine_similarity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "avg_cosine_similarity"])
        for i, val in enumerate(avg_cos_sim):
            writer.writerow([i, val])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(num_layers), avg_cos_sim, marker='o', linewidth=1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Cosine Similarity (all tokens)")
    ax.set_title(f"{model_name} average token-pair cosine similarity vs depth")
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.savefig(model_dir / "avg_cosine_similarity_vs_depth.png")
    plt.close(fig)

    logmeans = []
    eigs_by_layer = {}
    powerlaw_fit_rows = []
    layer_step = 3 if num_layers <= 24 else 5
    selected_layers = list(range(0, num_layers, layer_step))
    if (num_layers - 1) not in selected_layers:
        selected_layers.append(num_layers - 1)
    
    for i in range(len(accs)):
        acc = accs[i]
        if acc is None:
            continue
        try:
            cov = acc.covariance()
            eigs = eigvals_sorted(cov, cfg.eps)
            plot_loglog_eigs(eigs, model_dir / f"layer_{i:02d}_eigs.png", f"{model_name} layer {i} eigs")
            slope, intercept, r2 = fit_powerlaw_loglog(eigs)
            powerlaw_fit_rows.append([i, slope, intercept, r2, int(eigs.shape[0])])
            if i in selected_layers:
                eigs_by_layer[i] = eigs.cpu() 
            logmeans.append(float(torch.log(eigs).mean().item()))
        except ValueError as e:
            print(f"\n[Warning] Skipping layer {i} spectra calculation: {e}")
            logmeans.append(float('nan'))
            
        accs[i] = None 
        del acc, cov, eigs
        free_memory() 

    fig, ax = plt.subplots(figsize=(7, 4))
    max_n = None
    for i in selected_layers:
        eigs = eigs_by_layer.get(i)
        if eigs is None:
            continue
        n = eigs.shape[0]
        max_n = n
        xs = np.arange(1, n + 1)
        ys = eigs.numpy()
        ax.plot(_piecewise_log2_x(xs), ys, linewidth=1.0, label=f"Layer {i}")
    ax.set_yscale("log")
    if max_n is not None:
        _set_piecewise_log2_xaxis(ax, max_n)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"{model_name} eigs by layer (every {layer_step})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(model_dir / "layer_eigs_overview.png")
    plt.close(fig)

    with open(model_dir / "layer_log_eig_mean.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "log_eig_mean"])
        for i, v in enumerate(logmeans):
            writer.writerow([i, v])

    with open(model_dir / "layer_powerlaw_fit.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "slope", "intercept", "r2", "num_points"])
        for row in powerlaw_fit_rows:
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(range(len(logmeans))), logmeans, marker="o", linewidth=1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean log eigenvalue")
    ax.set_title(f"{model_name} log-eig mean vs depth")
    fig.tight_layout()
    fig.savefig(model_dir / "log_eig_mean_vs_depth.png")
    plt.close(fig)

    # --- 5. 后续处理：计算 Two-NN 内在维度 ---
    print(f"[{model_name}] Processing Two-NN intrinsic dimension...")
    intrinsic_dims = []
    for l in range(num_layers):
        if twonn_layer_counts[l] == 0:
            intrinsic_dims.append(float('nan'))
            continue
        vectors = torch.cat(layer_samples[l], dim=0)   
        if vectors.shape[0] < 50:
            print(f"[Warning] Layer {l} has only {vectors.shape[0]} tokens, skipping")
            intrinsic_dims.append(float('nan'))
            continue
        d_est = estimate_intrinsic_dimension_twonn(
            vectors,
            subsample=min(cfg.embedding_sample_size, vectors.shape[0])
        )
        intrinsic_dims.append(d_est)
        layer_samples[l] = None
        free_memory()

    with open(model_dir / "twonn_intrinsic_dim.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "intrinsic_dimension"])
        for i, d in enumerate(intrinsic_dims):
            writer.writerow([i, d])

    fig, ax = plt.subplots(figsize=(7, 4))
    layers = list(range(num_layers))
    if any(not np.isnan(d) for d in intrinsic_dims):
        ax.plot(layers, intrinsic_dims, marker='o', linewidth=1.0)
        ax.set_ylabel("Estimated intrinsic dimension (Two-NN)")
        ax.set_xlabel("Layer")
        ax.set_title(f"{model_name} – Two-NN intrinsic dimension vs depth")
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        fig.savefig(model_dir / "twonn_intrinsic_dim_vs_depth.png")
    else:
        print(f"[Warning] No valid intrinsic dimension estimates for {model_name}")
    plt.close(fig)

def analyze_model(model_name: str, texts, cfg: Config, device: torch.device) -> None:
    def _download_model_from_mirror(mirror_base: str, model_name: str) -> Path:
        print(f"Downloading model from mirror: {mirror_base} for {model_name}")
        local_dir = snapshot_download(
            repo_id=model_name,
            repo_type="model",
            endpoint=mirror_base,
        )
        return Path(local_dir)

    print(f"Loading model: {model_name}")
    if cfg.model_mirror_base:
        local_dir = _download_model_from_mirror(cfg.model_mirror_base, model_name)
        tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
    else:
        raise ValueError("model_mirror_base must be set")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batches = build_batches(tokenizer, texts, cfg.batch_size, cfg.max_length, cfg.num_batches)
    print(f"Prepared {len(batches)} batches for {model_name}")

    if cfg.model_mirror_base:
        local_dir = _download_model_from_mirror(cfg.model_mirror_base, model_name)
        model = AutoModelForCausalLM.from_pretrained(
            str(local_dir), 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
            device_map="auto"  
        )
    else:
        raise ValueError("model_mirror_base must be set")

    model.eval()
    model_dir = cfg.output_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    target_device = model.device

    try:
        analyze_embedding_stats(model, model_name, cfg, model_dir, target_device)
    except Exception as e:
        print(f"\n[Warning] Embedding stats failed for {model_name}: {e}")

    blocks = _get_blocks(model)
    hidden_size = _get_hidden_size(model.config)
    if hidden_size is None:
        raise AttributeError("Model config missing hidden size")

    try:
        # Pass merged now
        analyze_effective_ranks_and_entropy(blocks, hidden_size, model_name, model_dir, target_device)
    except Exception as e:
        print(f"\n[Warning] Effective ranks failed for {model_name}: {e}")

    try:
        analyze_hidden_states_and_twonn(
            model,
            batches,
            hidden_size,
            len(blocks),
            device,
            cfg,
            model_name,
            model_dir,
        )
    except Exception as e:
        print(f"\n[Warning] Combined Hidden States and Two-NN analysis failed for {model_name}: {e}")

    print(f"Releasing memory allocated for {model_name}...")
    del model
    del tokenizer
    del batches
    free_memory()

# ================================
# Main: load data and run for each model
# ================================
def load_texts():
    local_dataset_path = "dataset-wikitext-2-raw-v1"
    print(f"Loading dataset from local disk: {local_dataset_path}")
    dataset = load_from_disk(local_dataset_path)

    if hasattr(dataset, "keys"):
        texts = []
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            if "text" in split_data.column_names:
                texts.extend(split_data["text"])
    else:
        texts = dataset["text"] if "text" in dataset.column_names else []

    texts = [t for t in texts if len(t.strip()) > 30]
    print(f"Loaded {len(texts)} texts from local dataset")
    return texts

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    texts = load_texts()

    for model_name in cfg.model_names:
        analyze_model(model_name, texts, cfg, device)

    print("Done. See results/<model> for outputs.")