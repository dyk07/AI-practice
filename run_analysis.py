#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer-based LLM stats analysis

This script computes:
- Embedding covariance and eigen spectrum
- Effective rank (99% energy) for Q, K, V, and out projection weights
- Hidden state covariance and eigen spectrum after parameter-free LayerNorm
- Mean log eigenvalue per layer and plots

Results are saved under results/<model>.
"""

import os
import ctypes
import csv
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
# Limit OpenMP threads to prevent SVD memory bloat on 16 CPUs
os.environ["OMP_NUM_THREADS"] = "8" 
torch.set_num_threads(8)

def free_memory():
    """Forces Python and the OS to reclaim memory."""
    gc.collect()
    try:
        # Force glibc to return freed memory back to the OS
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
# ================================
# Configuration
# ================================
@dataclass
class Config:
    model_names = ["Qwen/Qwen3-32B"]
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-2-raw-v1"
    dataset_mirror: Optional[str] = "https://hf-mirror.com"
    model_mirror_base: Optional[str] = "https://hf-mirror.com"
    max_length = 128
    batch_size = 8
    num_batches = 128
    embedding_sample_size = 10000  # set an int to speed up embedding covariance
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
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=gen)[:max_rows]
    return x[idx]


def covariance_from_rows(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    return (x.T @ x) / (x.shape[0] - 1)


class CovAccumulator:
    def __init__(self, dim: int, dtype: torch.dtype):
        self.dim = dim
        self.dtype = dtype
        self.count = 0
        self.sum = torch.zeros(dim, dtype=dtype)
        self.sum_xtx = torch.zeros(dim, dim, dtype=dtype)

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
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(_piecewise_log2_x(xs), ys, linewidth=1.0)
    ax.set_yscale("log")
    _set_piecewise_log2_xaxis(ax, n)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def effective_rank_99(weight: torch.Tensor) -> tuple[int, float]:
    s = torch.linalg.svdvals(weight.float())
    energy = torch.cumsum(s * s, dim=0)
    total = energy[-1]
    k = int(torch.searchsorted(energy, 0.99 * total).item() + 1)
    denom = min(weight.shape)
    return k, k / denom


def effective_rank_entropy(weight: torch.Tensor) -> tuple[int, float]:
    s = torch.linalg.svdvals(weight.float())
    p = s[s > 1e-10] / s[s > 1e-10].sum()
    erank = torch.exp(-torch.sum(p * torch.log(p))) if p.numel() > 0 else torch.tensor(0.0)
    k_float = erank.item()
    k = int(round(k_float))
    denom = min(weight.shape)
    return k, k_float / denom


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
        w_qkv = attn.c_attn.weight.detach().cpu()
        w_q = w_qkv[:, :hidden_size]
        w_k = w_qkv[:, hidden_size : 2 * hidden_size]
        w_v = w_qkv[:, 2 * hidden_size :]
        w_o = attn.c_proj.weight.detach().cpu()
        return w_q, w_k, w_v, w_o

    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    o_proj = getattr(attn, "o_proj", None)
    if all([q_proj, k_proj, v_proj, o_proj]):
        return (
            q_proj.weight.detach().cpu(),
            k_proj.weight.detach().cpu(),
            v_proj.weight.detach().cpu(),
            o_proj.weight.detach().cpu(),
        )
    raise AttributeError("Unsupported attention layout")


def analyze_embedding_stats(model, model_name: str, cfg: Config, model_dir: Path) -> None:
    wte_full = model.get_input_embeddings().weight.detach().cpu()
    wte_full = layernorm_no_params(wte_full, cfg.eps)
    wte = maybe_sample_rows(wte_full, cfg.embedding_sample_size)
    cov_emb = covariance_from_rows(wte)
    eig_emb = eigvals_sorted(cov_emb, cfg.eps)
    plot_loglog_eigs(eig_emb, model_dir / "embedding_eigs.png", f"{model_name} embedding eigs")
    emb_logmean = float(torch.log(eig_emb).mean().item())

    with open(model_dir / "embedding_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["log_eig_mean"])
        writer.writerow([emb_logmean])

    wte_f = wte_full.float()
    norms = torch.linalg.norm(wte_f, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=cfg.eps)
    wte_unit = wte_f / norms
    vocab_size = wte_unit.shape[0]
    sample_size = min(4096, vocab_size)
    if vocab_size < 2:
        nn_mean_cos = float("nan")
        mean_pairwise_cos = float("nan")
    else:
        gen = torch.Generator().manual_seed(42)
        sample_idx = torch.randperm(vocab_size, generator=gen)[:sample_size]
        sample = wte_unit[sample_idx]
        sims = sample @ wte_unit.T
        sims[torch.arange(sample_size), sample_idx] = -1.0
        nn_mean_cos = float(sims.max(dim=1).values.mean().item())
        sum_vec = wte_unit.double().sum(dim=0)
        sum_sq = float((sum_vec @ sum_vec).item())
        mean_pairwise_cos = (sum_sq - vocab_size) / (vocab_size * (vocab_size - 1))

    with open(model_dir / "embedding_similarity_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_size", "nn_mean_cosine", "mean_pairwise_cosine"])
        writer.writerow([sample_size, nn_mean_cos, mean_pairwise_cos])


def analyze_effective_ranks(blocks, hidden_size: int, model_name: str, model_dir: Path) -> None:
    ranks = {"q": [], "k": [], "v": [], "o": []}
    dims = {"q": "", "k": "", "v": "", "o": ""}

    for i, block in enumerate(tqdm(blocks, desc=f"SVD ranks ({model_name})")):
        w_q, w_k, w_v, w_o = _get_attn_weights(block, hidden_size)

        if i == 0:
            dims["q"] = f"{w_q.shape[0]}x{w_q.shape[1]}"
            dims["k"] = f"{w_k.shape[0]}x{w_k.shape[1]}"
            dims["v"] = f"{w_v.shape[0]}x{w_v.shape[1]}"
            dims["o"] = f"{w_o.shape[0]}x{w_o.shape[1]}"

        _, rq = effective_rank_99(w_q)
        _, rk = effective_rank_99(w_k)
        _, rv = effective_rank_99(w_v)
        _, ro = effective_rank_99(w_o)

        ranks["q"].append(rq)
        ranks["k"].append(rk)
        ranks["v"].append(rv)
        ranks["o"].append(ro)

    with open(model_dir / "effective_ranks.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "q", "k", "v", "o"])
        for i in range(len(ranks["q"])):
            writer.writerow([i, ranks["q"][i], ranks["k"][i], ranks["v"][i], ranks["o"][i]])

    plt.figure(figsize=(7, 4.5))
    layers = list(range(len(ranks["q"])))
    plt.plot(layers, ranks["q"], label=f"Q ({dims['q']})")
    plt.plot(layers, ranks["k"], label=f"K ({dims['k']})")
    plt.plot(layers, ranks["v"], label=f"V ({dims['v']})")
    plt.plot(layers, ranks["o"], label=f"Out ({dims['o']})")
    plt.xlabel("Layer")
    plt.ylabel("Normalized effective rank (99%)")
    plt.title(f"{model_name} effective rank vs depth")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    plt.savefig(model_dir / "effective_rank_vs_depth.png", bbox_inches='tight')
    plt.close()


def analyze_effective_rank_entropy(blocks, hidden_size: int, model_name: str, model_dir: Path) -> None:
    ranks = {"q": [], "k": [], "v": [], "o": []}
    dims = {"q": "", "k": "", "v": "", "o": ""}

    for i, block in enumerate(tqdm(blocks, desc=f"SVD ranks entropy ({model_name})")):
        w_q, w_k, w_v, w_o = _get_attn_weights(block, hidden_size)

        if i == 0:
            dims["q"] = f"{w_q.shape[0]}x{w_q.shape[1]}"
            dims["k"] = f"{w_k.shape[0]}x{w_k.shape[1]}"
            dims["v"] = f"{w_v.shape[0]}x{w_v.shape[1]}"
            dims["o"] = f"{w_o.shape[0]}x{w_o.shape[1]}"

        _, rq = effective_rank_entropy(w_q)
        _, rk = effective_rank_entropy(w_k)
        _, rv = effective_rank_entropy(w_v)
        _, ro = effective_rank_entropy(w_o)

        ranks["q"].append(rq)
        ranks["k"].append(rk)
        ranks["v"].append(rv)
        ranks["o"].append(ro)

    with open(model_dir / "effective_rank_entropy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "q", "k", "v", "o"])
        for i in range(len(ranks["q"])):
            writer.writerow([i, ranks["q"][i], ranks["k"][i], ranks["v"][i], ranks["o"][i]])

    plt.figure(figsize=(7, 4.5))
    layers = list(range(len(ranks["q"])))
    plt.plot(layers, ranks["q"], label=f"Q ({dims['q']})")
    plt.plot(layers, ranks["k"], label=f"K ({dims['k']})")
    plt.plot(layers, ranks["v"], label=f"V ({dims['v']})")
    plt.plot(layers, ranks["o"], label=f"Out ({dims['o']})")
    plt.xlabel("Layer")
    plt.ylabel("Normalized effective rank entropy")
    plt.title(f"{model_name} effective rank entropy vs depth")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    plt.savefig(model_dir / "effective_rank_entropy_vs_depth.png", bbox_inches='tight')
    plt.close()
#for gpu use
def analyze_hidden_state_spectra(
    model,
    batches,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    cfg: Config,
    model_name: str,
    model_dir: Path,
) -> None:
    # Accumulators (kept on CPU, saves GPU VRAM)
    accs = [CovAccumulator(hidden_size, cfg.cov_dtype) for _ in range(num_layers)]
    sum_unit_vecs = [torch.zeros(hidden_size, dtype=cfg.cov_dtype, device='cpu') for _ in range(num_layers)]
    token_counts = [0] * num_layers

    # REVISION: Get the device where the model's first layer is located
    target_device = model.device

    for batch in tqdm(batches, desc=f"Hidden states ({model_name})"):
        # REVISION: Move batch to target_device instead of the generic 'device'
        batch = {k: v.to(target_device) for k, v in batch.items()}
        attn_mask = batch.get("attention_mask")
        token_mask = attn_mask.reshape(-1).bool() if attn_mask is not None else None

        with torch.inference_mode():
            outputs = model(**batch, output_hidden_states=True, use_cache=False)

        hidden_list = list(outputs.hidden_states)
        del outputs

        for i in range(1, len(hidden_list)):
            h = hidden_list[i]                     
            idx = i - 1

            x = h.reshape(-1, h.shape[-1])
            if token_mask is not None:
                # Need to make sure token_mask is on the same device as h
                x = x[token_mask.to(x.device)]
                if x.numel() == 0:
                    hidden_list[i] = None
                    continue

            x = layernorm_no_params(x, cfg.eps)

            # Move to CPU directly after computation to save GPU memory
            x_cpu = x.cpu().float() 
            norm = torch.norm(x_cpu, dim=-1, keepdim=True).clamp(min=cfg.eps)
            x_unit = x_cpu / norm
            
            sum_unit_vecs[idx] += x_unit.sum(dim=0)
            token_counts[idx] += x_unit.shape[0]

            accs[idx].update(x_cpu)
            hidden_list[i] = None

        del hidden_list
        del batch

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

    plt.figure(figsize=(7, 4))
    plt.plot(range(num_layers), avg_cos_sim, marker='o', linewidth=1.0)
    plt.xlabel("Layer")
    plt.ylabel("Average Cosine Similarity (all tokens)")
    plt.title(f"{model_name} average token-pair cosine similarity vs depth")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(model_dir / "avg_cosine_similarity_vs_depth.png")
    plt.close()

    # 特征值谱计算（与原代码相同，略）
    logmeans = []
    eigs_by_layer = {}
    layer_step = 3 if num_layers <= 24 else 5
    selected_layers = list(range(0, num_layers, layer_step))
    if (num_layers - 1) not in selected_layers:
        selected_layers.append(num_layers - 1)
    
    for i in tqdm(range(len(accs)), desc=f"Eig spectra ({model_name})"):
        acc = accs[i]
        if acc is None:
            continue
        try:
            cov = acc.covariance()
            eigs = eigvals_sorted(cov, cfg.eps)
            plot_loglog_eigs(eigs, model_dir / f"layer_{i:02d}_eigs.png", f"{model_name} layer {i} eigs")
            if i in selected_layers:
                eigs_by_layer[i] = eigs # Small enough to keep in memory
            logmeans.append(float(torch.log(eigs).mean().item()))
        except ValueError as e:
            print(f"\n[Warning] Skipping layer {i} spectra calculation: {e}")
            logmeans.append(float('nan'))
            
        # CRITICAL FIX: Destroy the accumulator immediately!
        accs[i] = None 
        free_memory() # Reclaim the ~104MB chunk per layer instantly

    # 后续绘图保存部分不变...
    plt.figure(figsize=(7, 4))
    max_n = None
    for i in selected_layers:
        eigs = eigs_by_layer.get(i)
        if eigs is None:
            continue
        n = eigs.shape[0]
        max_n = n
        xs = np.arange(1, n + 1)
        ys = eigs.cpu().numpy()
        plt.plot(_piecewise_log2_x(xs), ys, linewidth=1.0, label=f"Layer {i}")
    ax = plt.gca()
    ax.set_yscale("log")
    if max_n is not None:
        _set_piecewise_log2_xaxis(ax, max_n)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"{model_name} eigs by layer (every {layer_step})")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(model_dir / "layer_eigs_overview.png")
    plt.close()

    with open(model_dir / "layer_log_eig_mean.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "log_eig_mean"])
        for i, v in enumerate(logmeans):
            writer.writerow([i, v])

    plt.figure(figsize=(7, 4))
    plt.plot(list(range(len(logmeans))), logmeans, marker="o", linewidth=1.0)
    plt.xlabel("Layer")
    plt.ylabel("Mean log eigenvalue")
    plt.title(f"{model_name} log-eig mean vs depth")
    plt.tight_layout()
    plt.savefig(model_dir / "log_eig_mean_vs_depth.png")
    plt.close()

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
        # REVISION: Use bfloat16 and device_map="auto" for large models
        model = AutoModelForCausalLM.from_pretrained(
            str(local_dir), 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
            device_map="auto"  # Automatically spreads across available GPUs
        )
    else:
        raise ValueError("model_mirror_base must be set")

    # REVISION: Remove model.to(device) as device_map="auto" handles placement automatically
    model.eval()

    model_dir = cfg.output_dir / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        analyze_embedding_stats(model, model_name, cfg, model_dir)
    except Exception as e:
        print(f"\n[Warning] Embedding stats failed for {model_name}: {e}")

    blocks = _get_blocks(model)
    hidden_size = _get_hidden_size(model.config)
    if hidden_size is None:
        raise AttributeError("Model config missing hidden size")

    try:
        analyze_effective_ranks(blocks, hidden_size, model_name, model_dir)
    except Exception as e:
        print(f"\n[Warning] Effective ranks failed for {model_name}: {e}")

    try:
        analyze_effective_rank_entropy(blocks, hidden_size, model_name, model_dir)
    except Exception as e:
        print(f"\n[Warning] Effective rank entropy failed for {model_name}: {e}")

    try:
        analyze_hidden_state_spectra(
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
        print(f"\n[Warning] Hidden state spectra failed for {model_name}: {e}")

    print(f"Releasing memory allocated for {model_name}...")
    del model
    del tokenizer
    del batches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================================
# Main: load data and run for each model
# ================================
def load_texts():
    local_dataset_path = "dataset-wikitext-2-raw-v1"
    print(f"Loading dataset from local disk: {local_dataset_path}")
    dataset = load_from_disk(local_dataset_path)

    if hasattr(dataset, "keys"):  # DatasetDict
        texts = []
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            if "text" in split_data.column_names:
                texts.extend(split_data["text"])
    else:  # Dataset
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