"""
Efficiency measurement script for Table 2 (cost/complexity), covering A2 / R1.2 / R3.5.

Setup:
    pip install fvcore torchvision transformers

Run:
    python measure_efficiency.py              # full table (ours + baselines)
    python measure_efficiency.py --ours-only    # our model only (faster, no BERT download)
"""

import argparse
import time
import torch
import config
from base_models_refactored_v1 import MultimodalFusion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = config.get_max_token_length()


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, inputs)
        fca.unsupported_ops_warnings(False)
        return fca.total() / 1e9
    except Exception as e:
        return f"FLOP count failed: {e}"


def measure_latency_and_memory(model, make_input, batch_sizes, n_warmup=15, n_iters=50):
    model = model.to(DEVICE).eval()
    out = {}
    for bs in batch_sizes:
        inputs = make_input(bs, DEVICE)
        with torch.no_grad():
            for _ in range(n_warmup):
                model(*inputs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(n_iters):
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(*inputs)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        mean_s = sum(times) / len(times)
        out[bs] = {
            "latency_ms_total": mean_s * 1000,
            "latency_ms_per_sample": mean_s * 1000 / bs,
            "peak_mem_MB": (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else None,
        }
    return out


def run_all(name, model, make_input, batch_sizes=(1, 32)):
    print(f"\n=== {name} ===")
    params = count_params(model)
    print(f"Params (M): {params:.2f}")
    flops = count_flops(model.to(DEVICE).eval(), make_input(1, DEVICE))
    print(f"FLOPs (G) @ bs=1: {flops}")
    perf = measure_latency_and_memory(model, make_input, batch_sizes)
    for bs, stats in perf.items():
        mem_str = f"{stats['peak_mem_MB']:.1f}MB" if stats["peak_mem_MB"] is not None else "N/A"
        print(
            f"  bs={bs}: total={stats['latency_ms_total']:.2f}ms  "
            f"per-sample={stats['latency_ms_per_sample']:.2f}ms  peak_mem={mem_str}"
        )
    return {"params_M": params, "flops_G": flops, "perf": perf}


def your_input_fn(bs, device):
    images = torch.randn(bs, 3, 224, 224, device=device)
    texts = torch.randint(1, config.get_vocab_size(), (bs, SEQ_LEN), device=device)
    return ((images, texts),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-only", action="store_true", help="Benchmark MultimodalFusion only")
    parser.add_argument("--n-iters", type=int, default=50, help="Latency benchmark iterations")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Dataset mode: {config.DATASET_MODE}")
    print(f"SEQ_LEN: {SEQ_LEN}, vocab_size: {config.get_vocab_size()}")

    your_model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()["num_heads"],
        num_layers=config.get_current_config()["num_layers"],
    )

    def run_with_iters(name, model, make_input, batch_sizes=(1, 32)):
        global measure_latency_and_memory
        _orig = measure_latency_and_memory

        def _patched(model, make_input, batch_sizes, n_warmup=15, n_iters=50):
            return _orig(model, make_input, batch_sizes, n_warmup=n_warmup, n_iters=args.n_iters)

        measure_latency_and_memory = _patched
        try:
            return run_all(name, model, make_input, batch_sizes)
        finally:
            measure_latency_and_memory = _orig

    ours = run_with_iters("Your model (full, with co-attention)", your_model, your_input_fn)

    if args.ours_only:
        print("\n(--ours-only: skipping ResNet-50 and ClinicalBERT baselines)")
        raise SystemExit(0)

    import torchvision.models as tvm
    from transformers import AutoModel

    resnet = tvm.resnet50(weights=None)

    def resnet_input_fn(bs, device):
        return (torch.randn(bs, 3, 224, 224, device=device),)

    resnet_stats = run_all("Baseline: ResNet-50", resnet, resnet_input_fn)

    clinicalbert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def bert_input_fn(bs, device):
        ids = torch.randint(0, clinicalbert.config.vocab_size, (bs, SEQ_LEN), device=device)
        mask = torch.ones(bs, SEQ_LEN, device=device)
        return (ids, mask)

    bert_stats = run_all("Baseline: ClinicalBERT", clinicalbert, bert_input_fn)

    combined_params = resnet_stats["params_M"] + bert_stats["params_M"]
    print("\n=== Baseline combined (ResNet-50 + ClinicalBERT, additive) ===")
    print(f"Params (M): {combined_params:.2f}")

    r_perf = resnet_stats["perf"]
    b_perf = bert_stats["perf"]
    for bs in (1, 32):
        lat = r_perf[bs]["latency_ms_per_sample"] + b_perf[bs]["latency_ms_per_sample"]
        mem = None
        if r_perf[bs]["peak_mem_MB"] is not None and b_perf[bs]["peak_mem_MB"] is not None:
            mem = r_perf[bs]["peak_mem_MB"] + b_perf[bs]["peak_mem_MB"]
        mem_str = f"{mem:.1f}MB" if mem is not None else "N/A"
        print(f"  bs={bs}: per-sample latency (sum)={lat:.2f}ms  peak_mem (sum)={mem_str}")

    r_flops = resnet_stats["flops_G"]
    b_flops = bert_stats["flops_G"]
    if isinstance(r_flops, (int, float)) and isinstance(b_flops, (int, float)):
        print(f"Combined FLOPs (G) @ bs=1: {r_flops + b_flops:.2f}")
    else:
        print("Combined FLOPs: sum of individual baseline FLOPs above")

    print("\n=== Ours vs combined baseline (quick summary) ===")
    o_perf = ours["perf"]
    print(f"Params: ours {ours['params_M']:.2f}M vs baseline {combined_params:.2f}M")
    if isinstance(ours["flops_G"], (int, float)) and isinstance(r_flops, (int, float)) and isinstance(b_flops, (int, float)):
        print(f"FLOPs (G): ours {ours['flops_G']:.2f} vs baseline {r_flops + b_flops:.2f}")
    print(
        f"Latency bs=1 (ms/sample): ours {o_perf[1]['latency_ms_per_sample']:.2f} "
        f"vs baseline {r_perf[1]['latency_ms_per_sample'] + b_perf[1]['latency_ms_per_sample']:.2f}"
    )
    print(
        f"Latency bs=32 (ms/sample): ours {o_perf[32]['latency_ms_per_sample']:.2f} "
        f"vs baseline {r_perf[32]['latency_ms_per_sample'] + b_perf[32]['latency_ms_per_sample']:.2f}"
    )
