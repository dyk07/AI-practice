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
    batch_size = 256
    num_batches = 64
    embedding_sample_size = 10000 
    cov_dtype = torch.float32
    eps = 1e-6
    to_device = "cuda" if torch.cuda.is_available() else "cpu"
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
    def __init__(self, dim: int, dtype: torch.dtype, device: torch.device):
        self.dim = dim
        self.dtype = dtype
        self.count = 0
        # Initialize directly on the target device (GPU) to utilize 320GB VRAM
        self.sum = torch.zeros(dim, dtype=dtype, device=device)
        self.sum_xtx = torch.zeros(dim, dim, dtype=dtype, device=device)

    def update(self, x: torch.Tensor) -> None:
        """Everything remains on the assigned GPU, maximizing speed and removing CPU memory overhead."""
        x_calc = x.to(self.dtype).to(self.sum.device)
        self.sum += x_calc.sum(dim=0)
        self.sum_xtx += x_calc.T @ x_calc
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
    ys = eigvals.detach().cpu().numpy().astype(float)
    xs = np.arange(1, ys.shape[0] + 1, dtype=float)
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
    w_gpu = weight.to(device, dtype=torch.float32)
    s = torch.linalg.svdvals(w_gpu)
    
    energy = torch.cumsum(s * s, dim=0)
    total = energy[-1]
    k_99 = int(torch.searchsorted(energy, 0.99 * total).item() + 1)
    
    p = s[s > 1e-10] / s[s > 1e-10].sum()
    erank = torch.exp(-torch.sum(p * torch.log(p))) if p.numel() > 0 else torch.tensor(0.0)
    k_ent = int(round(erank.item()))
    
    denom = min(weight.shape)
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
        
        del w_q, w_k, w_v, w_o
        free_memory()
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
    if y is None:
        y = x
    x2 = torch.mean(x ** 2, dim=1, keepdim=True)   
    y2 = torch.mean(y ** 2, dim=1, keepdim=True)   
    xy = torch.mm(x, y.T) / x.shape[1]             
    dist2 = x2 + y2.T - 2 * xy
    return dist2.clamp(min=0.0)

def estimate_intrinsic_dimension_twonn_chunked(vectors_cpu: torch.Tensor, device: torch.device, chunk_size: int = 2048) -> float:
    """
    分块自适应 Two-NN 估算算法：
    将全量数据常驻 CPU，每次只将一小块 (chunk_size) 扔上 GPU 与全量数据计算距离，
    在 GPU 内通过 topk 直接过滤出 r1 和 r2，彻底避免 N x N 显存爆炸。
    """
    N, D = vectors_cpu.shape
    if N <= 2:
        return float('nan')

    # 用于存放所有样本的 r1 和 r2
    r1_list = []
    r2_list = []

    # 将全量数据转为 float32 用于精确距离计算
    vectors_cpu = vectors_cpu.float()
    
    # 提前计算全量数据的平方和，并转成 GPU 上的列向量，方便利用矩阵乘法加速：||x - y||^2 = ||x||^2 + ||y||^2 - 2xy
    v_sq_cpu = torch.mean(vectors_cpu ** 2, dim=1, keepdim=True)

    # 逐块作为 Anchor 节点去查询全量节点
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        curr_chunk = vectors_cpu[start_idx:end_idx].to(device) # [chunk, D]
        curr_sq = v_sq_cpu[start_idx:end_idx].to(device)       # [chunk, 1]

        # 分批将全量数据搬运/保持在 GPU 加速矩阵乘法 (如果全量数据很大，这里可以进一步分块，但当前内存足够)
        all_vectors_gpu = vectors_cpu.to(device)
        all_sq_gpu = v_sq_cpu.to(device)

        # 核心：计算当前 chunk 与全量数据的 MSE 距离矩阵 [chunk, N]
        xy = torch.mm(curr_chunk, all_vectors_gpu.T) / D
        dist2 = curr_sq + all_sq_gpu.T - 2 * xy
        dist2 = dist2.clamp(min=0.0)

        # 排除掉自身（对角线位置距离设为无穷大）
        global_idx = torch.arange(start_idx, end_idx, device=device)
        dist2[torch.arange(end_idx - start_idx, device=device), global_idx] = float('inf')

        # 在 GPU 上直接找出最近的 3 个邻居 (因为自身可能由于微小浮点误差排在前面，取 3 个最稳妥)
        # 排序：找最小的距离
        topk_val, _ = torch.topk(dist2, k=3, largest=False, dim=1)

        # 提取 r1 和 r2 的平方根（真实的物理距离）
        # 第一近邻 r1, 第二近邻 r2
        r1_chunk = torch.sqrt(topk_val[:, 0])
        r2_chunk = torch.sqrt(topk_val[:, 1])

        r1_list.append(r1_chunk.cpu())
        r2_list.append(r2_chunk.cpu())

        del curr_chunk, curr_sq, all_vectors_gpu, all_sq_gpu, dist2, topk_val
        if start_idx % (chunk_size * 5) == 0:
            torch.cuda.empty_cache()

    # 拼接全量结果进行统计分析
    r1 = torch.cat(r1_list, dim=0)
    r2 = torch.cat(r2_list, dim=0)

    mu = r2 / (r1 + 1e-12)
    mu = mu[mu > 1.0]

    if mu.numel() < 10:
        return float('nan')

    mu_sorted = torch.sort(mu)[0].numpy()
    Nmu = len(mu_sorted)
    F = np.arange(1, Nmu + 1) / Nmu
    log_mu = np.log(mu_sorted)
    log_one_minus_F = np.log(1 - F + 1e-12)

    # 线性回归拟合斜率
    trim = max(1, int(0.1 * Nmu))
    X = log_mu[:-trim] if Nmu > trim else log_mu
    y = log_one_minus_F[:-trim]
    if len(X) < 5:
        return float('nan')

    coeffs = np.polyfit(X, y, 1)
    return float(-coeffs[0])

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
    target_device = device

    # 1. 频谱累加器 (大小固定，保持在 GPU)
    accs = [CovAccumulator(hidden_size, cfg.cov_dtype, device=target_device) for _ in range(num_layers)]
    sum_unit_vecs = [torch.zeros(hidden_size, dtype=cfg.cov_dtype, device=target_device) for _ in range(num_layers)]
    token_counts = [0] * num_layers

    # 【最大限制扩大】：这里将上限扩大到 200,000 个 token（或者可以设为 500000 甚至不设上限）
    # 512 batch_size * 128 batches * 128 length 理论最大有 8M 多个 token。
    # 建议设为 100,000 或 200,000，这已经具备极强的统计学代表性，且 CPU 内存完全撑得住。
    max_tokens_per_layer = 50000  
    layer_samples = [[] for _ in range(num_layers)]
    twonn_layer_counts = [0] * num_layers

    # --- 数据收集阶段 ---
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Hidden states & Two-NN ({model_name})")):
        batch_dev = {k: v.to(target_device) for k, v in batch.items()}
        attn_mask = batch_dev.get("attention_mask")
        
        with torch.inference_mode():
            outputs = model(**batch_dev, output_hidden_states=True, use_cache=False)
            hidden_list = list(outputs.hidden_states)
        
        del outputs
        del batch_dev

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

            # Two-NN 数据收集：全力满足至多 150,000 个样本，采集后即时转存 CPU
            need = max_tokens_per_layer - twonn_layer_counts[idx]
            if need > 0:
                h_twonn = layernorm_no_params(h_flat, cfg.eps)
                if h_flat.shape[0] > need:
                    idx_perm = torch.randperm(h_twonn.shape[0], device=target_device)[:need]
                    # Change from: h_sample = h_twonn.cpu().float()
                    h_sample = h_twonn.cpu().to(torch.bfloat16) # Cuts the 196GB RAM footprint entirely in half to 98GB!
                else:
                    h_sample = h_twonn.cpu().to(torch.bfloat16)
                layer_samples[idx].append(h_sample)
                twonn_layer_counts[idx] += h_sample.shape[0]

            # 频谱协方差收集 (GPU 原地累加)
            x_gpu = layernorm_no_params(h_flat, cfg.eps).float()
            norm = torch.norm(x_gpu, dim=-1, keepdim=True).clamp(min=cfg.eps)
            x_unit = x_gpu / norm
            
            sum_unit_vecs[idx] += x_unit.sum(dim=0)
            token_counts[idx] += x_unit.shape[0]
            accs[idx].update(x_gpu)
            
            del x_gpu, norm, x_unit, h
            hidden_list[i] = None

        del hidden_list
        if (batch_idx + 1) % 10 == 0:
            free_memory()

    # --- 统计余弦相似度（略，与原脚本一致） ---
    print(f"[{model_name}] Processing hidden state spectra & Two-NN metrics...")
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
    
    intrinsic_dims = []

    # --- 逐层流水线解算阶段 ---
    for i in range(num_layers):
        # A. 频谱计算
        acc = accs[i]
        if acc is not None:
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
            del acc, cov
        else:
            logmeans.append(float('nan'))

        # B. Two-NN 计算 【核心修改：调用全新的 Chunked 计算函数】
        if twonn_layer_counts[i] == 0:
            intrinsic_dims.append(float('nan'))
        else:
            # 1. 在 CPU 端拼接几万到十几万的大型张量
            vectors_cpu = torch.cat(layer_samples[i], dim=0)
            layer_samples[i] = None # 即刻释放列表引用
            
            # 2. 调用分块 KNN 函数，传入常驻的 target_device (cuda)
            d_est = estimate_intrinsic_dimension_twonn_chunked(
                vectors_cpu, 
                device=target_device, 
                chunk_size=2048
            )
            intrinsic_dims.append(d_est)
            del vectors_cpu

        if i % 3 == 0 or i == num_layers - 1:
            free_memory()

    # --- 4. 汇总图表绘制 ---
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
    target_device = device

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
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        device_map = "auto" if torch.cuda.is_available() else None
        
        model = AutoModelForCausalLM.from_pretrained(
            str(local_dir), 
            torch_dtype=model_dtype, 
            low_cpu_mem_usage=True,
            device_map=device_map
        )
    else:
        raise ValueError("model_mirror_base must be set")

    if device_map is None:
        model = model.to(target_device)
    else:
        if hasattr(model, "device"):
            target_device = model.device

    model.eval()
    model_dir = cfg.output_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        analyze_embedding_stats(model, model_name, cfg, model_dir, target_device)
    except Exception as e:
        print(f"\n[Warning] Embedding stats failed for {model_name}: {e}")

    blocks = _get_blocks(model)
    hidden_size = _get_hidden_size(model.config)
    if hidden_size is None:
        raise AttributeError("Model config missing hidden size")

    try:
        analyze_effective_ranks_and_entropy(blocks, hidden_size, model_name, model_dir, target_device)
    except Exception as e:
        print(f"\n[Warning] Effective ranks failed for {model_name}: {e}")

    try:
        analyze_hidden_states_and_twonn(
            model,
            batches,
            hidden_size,
            len(blocks),
            target_device,
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
    device = torch.device(cfg.to_device if cfg.to_device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    texts = load_texts()

    for model_name in cfg.model_names:
        analyze_model(model_name, texts, cfg, device)
        free_memory()

    print("Done. See results/<model> for outputs.")