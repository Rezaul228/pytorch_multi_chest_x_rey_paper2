#!/usr/bin/env python3
"""
20-step SMOKE TEST for the granularity-loss wiring in train_retrieval_v2_paper2.py.
NOT a real epoch, NOT a real training run: no checkpoint saved, no SLURM job
submitted. Runs exactly 20 real training steps (real forward pass, real
backward pass, real optimizer.step()) on real train data, and checks:
  - loss values (total + 4 weighted components: synergy/main/orthogonal/gran)
    at steps 1, 10, 20 -- all finite, no explosion
  - the NEW section_find_gate_weights/section_imp_gate_weights (synergy
    branch only) actually change value between step 1 and step 20
  - an ORIGINAL Paper 1 parameter (an existing gate weight + an image
    encoder conv weight) also still updates normally
"""

import os
import sys
import copy
import time

import torch
import torch.optim as optim

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from base_models_refactored_v1_paper2 import MultimodalFusion
from data_loader_v1_paper2 import IndianaDataLoader
from train_retrieval_v2_paper2 import EnhancedRetrievalTrainer
import config
from torch.utils.data import DataLoader as TorchDataLoader

N_STEPS = 20
MAX_SAMPLES_FOR_TEST = 6000  # comfortably more than N_STEPS * batch_size, avoids loading the full ~155,745-sample train split


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = config.get_default_batch_size()
    learning_rate = config.get_default_learning_rate()
    print(f"Batch size: {batch_size}  Learning rate: {learning_rate}")

    torch.manual_seed(42)

    print("\nLoading a small slice of real TRAIN data (capped, not the full ~155,745-sample split)...")
    data_loader = IndianaDataLoader(
        batch_size=batch_size,
        use_shards=True,
        shard_size=config.get_current_config()["shard_size"],
        shard_subfolder=config.DATASET_MODE,
    )
    data_loader.load_data(max_samples=MAX_SAMPLES_FOR_TEST, skip_processing=True)
    train_dataset = data_loader.get_data(max_samples=MAX_SAMPLES_FOR_TEST)
    print(f"Train dataset size for this smoke test: {len(train_dataset)}")
    print(f"Section boundaries loaded: {len(train_dataset.section_boundaries)}  fallback so far: {train_dataset.fallback_count}")

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("\nBuilding fresh MultimodalFusion (paper2) -- same as train_retrieval_v2_paper2.py's real "
          "training script, which always starts from a fresh random init, not a loaded checkpoint...")
    fusion_model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()["num_heads"],
        num_layers=config.get_current_config()["num_layers"],
    ).to(device)

    viz_dir = "/tmp/paper2_smoke_test_viz"
    trainer = EnhancedRetrievalTrainer(
        model=fusion_model,
        learning_rate=learning_rate,
        device=device,
        experiment_name="smoke_test_paper2",
        model_save_path=None,  # never used -- save_model() is never called in this smoke test
        viz_dir=viz_dir,
    )

    # Snapshot BEFORE any training step.
    section_find_before = trainer.model.synergy_branch.co_attn_layers[0].section_find_gate_weights.detach().clone()
    section_imp_before = trainer.model.synergy_branch.co_attn_layers[0].section_imp_gate_weights.detach().clone()
    original_gate_before = trainer.model.synergy_branch.co_attn_layers[0].global_text_gate_weights.detach().clone()
    conv_weight_before = trainer.model.image_encoder.conv_block_1[0].weight.detach().clone()

    print("\n" + "=" * 90)
    print(f"Running {N_STEPS} real training steps (real forward + backward + optimizer.step())")
    print("=" * 90)

    results = {}
    step_count = 0
    step_times = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    loop_start = time.perf_counter()
    for batch in train_loader:
        if step_count >= N_STEPS:
            break
        step_count += 1

        step_batch = {
            "images": batch["images"].to(device),
            "captions": batch["captions"].to(device),
            "findings_token_count": batch["findings_token_count"].to(device),
            "has_find": batch["has_find"].to(device),
            "has_imp": batch["has_imp"].to(device),
        }

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        total_loss, synergy_loss, difference_loss, orthogonal_loss, main_loss, gran_loss = trainer.train_step(step_batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        finite = all(
            torch.isfinite(torch.tensor(v)).item()
            for v in [total_loss, synergy_loss, difference_loss, orthogonal_loss, main_loss, gran_loss]
        )

        if step_count in (1, 10, 20):
            results[step_count] = {
                "total_loss": total_loss,
                "synergy_loss": synergy_loss,
                "main_loss": main_loss,
                "orthogonal_loss": orthogonal_loss,
                "gran_loss": gran_loss,
                "difference_loss": difference_loss,
                "finite": finite,
            }
            print(f"\nStep {step_count}:")
            print(f"  total_loss       = {total_loss:.6f}")
            print(f"  synergy_loss     = {synergy_loss:.6f}  (weight 0.65)")
            print(f"  main_loss        = {main_loss:.6f}  (weight 0.20)")
            print(f"  orthogonal_loss  = {orthogonal_loss:.6f}  (weight 0.15)")
            print(f"  gran_loss (NEW)  = {gran_loss:.6f}  (weight 0.10)")
            print(f"  difference_loss  = {difference_loss:.6f}  (not part of total_loss, unchanged from original)")
            print(f"  all finite: {finite}")
            print(f"  step wall time: {step_times[-1]:.4f}s")
        else:
            if not finite:
                print(f"Step {step_count}: NON-FINITE VALUE DETECTED -- total={total_loss} syn={synergy_loss} "
                      f"main={main_loss} ortho={orthogonal_loss} gran={gran_loss}")

    loop_total_time = time.perf_counter() - loop_start
    print(f"\nCompleted {step_count} steps.")

    print("\n" + "=" * 90)
    print("PER-STEP TIMING (real forward + backward + optimizer.step(), GPU-synchronized)")
    print("=" * 90)
    print(f"  all step times (s): {[round(t, 4) for t in step_times]}")
    print(f"  mean step time:   {sum(step_times)/len(step_times):.4f}s")
    print(f"  median step time: {sorted(step_times)[len(step_times)//2]:.4f}s")
    print(f"  min/max step time: {min(step_times):.4f}s / {max(step_times):.4f}s")
    print(f"  total loop time ({step_count} steps): {loop_total_time:.4f}s")
    # Exclude step 1 (often includes CUDA warmup/kernel compilation) for a steady-state estimate.
    steady_state_times = step_times[1:] if len(step_times) > 1 else step_times
    steady_state_mean = sum(steady_state_times) / len(steady_state_times)
    print(f"  mean step time EXCLUDING step 1 (steady-state, used for extrapolation): {steady_state_mean:.4f}s")

    # Snapshot AFTER training steps.
    section_find_after = trainer.model.synergy_branch.co_attn_layers[0].section_find_gate_weights.detach().clone()
    section_imp_after = trainer.model.synergy_branch.co_attn_layers[0].section_imp_gate_weights.detach().clone()
    original_gate_after = trainer.model.synergy_branch.co_attn_layers[0].global_text_gate_weights.detach().clone()
    conv_weight_after = trainer.model.image_encoder.conv_block_1[0].weight.detach().clone()

    print("\n" + "=" * 90)
    print("LOSS EXPLOSION / FINITENESS CHECK")
    print("=" * 90)
    totals = [results[k]["total_loss"] for k in sorted(results)]
    print(f"  total_loss at steps {sorted(results)}: {totals}")
    max_jump = max(abs(totals[i] - totals[i-1]) for i in range(1, len(totals))) if len(totals) > 1 else 0.0
    print(f"  max jump between reported steps: {max_jump:.6f}")
    all_finite_overall = all(results[k]["finite"] for k in results)
    print(f"  all reported steps finite: {all_finite_overall}")
    exploded = any(v > 1000 for k in results for key, v in results[k].items() if key != 'finite')
    print(f"  any loss component > 1000 (exploded): {exploded}")

    print("\n" + "=" * 90)
    print("NEW PARAMETER UPDATE CHECK (section_find_gate_weights / section_imp_gate_weights, synergy branch)")
    print("=" * 90)
    find_diff = (section_find_after - section_find_before).norm().item()
    imp_diff = (section_imp_after - section_imp_before).norm().item()
    print(f"  section_find_gate_weights: ||before - after|| = {find_diff:.8f}  changed={find_diff > 0}")
    print(f"  section_imp_gate_weights:  ||before - after|| = {imp_diff:.8f}  changed={imp_diff > 0}")

    print("\n" + "=" * 90)
    print("ORIGINAL PARAMETER UPDATE CHECK (sanity: nothing accidentally frozen)")
    print("=" * 90)
    orig_gate_diff = (original_gate_after - original_gate_before).norm().item()
    conv_diff = (conv_weight_after - conv_weight_before).norm().item()
    print(f"  global_text_gate_weights (existing Paper 1 gate): ||before - after|| = {orig_gate_diff:.8f}  changed={orig_gate_diff > 0}")
    print(f"  image_encoder.conv_block_1[0].weight:            ||before - after|| = {conv_diff:.8f}  changed={conv_diff > 0}")


if __name__ == "__main__":
    main()
