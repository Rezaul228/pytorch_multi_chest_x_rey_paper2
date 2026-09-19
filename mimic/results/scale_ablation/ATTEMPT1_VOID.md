# MIMIC data-scale ablation — ATTEMPT 1 (jobs 115242–115253): VOID

**Status: void as a scientific comparison. Never evaluated, and never to be
evaluated.** Weight files deleted 2026-09-20 to free ~8 GB; all training logs and
per-epoch metrics are retained as evidence.

## Why it is void

The runs used a fixed epoch count rather than a matched optimization budget:

| size | steps/epoch | epochs | total gradient steps | vs full-MIMIC budget |
|---|---|---|---|---|
| 2,553 | 20 | 400 | **8,000** | 26% |
| 10,000 | 79 | 150 | **11,850** | 39% |
| full MIMIC (reference) | 609 | 50 | **30,450** | 100% |

All 12 jobs completed cleanly (exit 0, no requeue, correct id filters, section
guard active on all 6 MG-G2L runs, zero boundary-fallback warnings, granularity
loss non-zero on every MG-G2L epoch line). The failure is in the design, not the
plumbing:

- **Neither arm converged at either size.**
- **5 of 6 runs at size 10,000 peaked in the final 10% of training** — the
  signature of a run cut off mid-climb.
- **Paper 1 seed 42 at size 10,000 collapsed**: best val R@1 **0.2102 at epoch
  81**, final **0.060**.
- Paper 1 was far noisier than MG-G2L at every size (at 10,000 the three Paper 1
  seeds ended at 0.576 / 0.060 / 0.909 against MG-G2L's 0.997 / 0.995 / 0.974).

Because MG-G2L largely converged and Paper 1 largely did not, any delta measured
on these checkpoints would mostly reflect **optimization speed against an
unconverged baseline**, not the architecture — and it would bias *against* the
pre-registered prediction by making the gain appear to grow at small scale for a
reason unrelated to data scale.

## Validation R@1 trajectory (training diagnostic, NOT a result)

Value at each 10% of the run, 10% → 100%:

| size | arm | seed | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2553 | Paper 1 | 17 | 0.0208 | 0.2617 | 0.0712 | 0.1613 | 0.2472 | 0.1730 | 0.3927 | 0.2415 | 0.4398 | **0.3550** |
| 2553 | Paper 1 | 42 | 0.0083 | 0.1492 | 0.5163 | 0.5302 | 0.3152 | 0.4393 | 0.5033 | 0.7160 | 0.7397 | **0.6300** |
| 2553 | Paper 1 | 123 | 0.0030 | 0.0108 | 0.0713 | 0.1570 | 0.1412 | 0.1303 | 0.2575 | 0.2090 | 0.2580 | **0.2627** |
| 2553 | MG-G2L | 17 | 0.0970 | 0.2475 | 0.2765 | 0.7443 | 0.6918 | 0.6840 | 0.7792 | 0.6425 | 0.7148 | **0.7267** |
| 2553 | MG-G2L | 42 | 0.0848 | 0.2108 | 0.2033 | 0.2867 | 0.4630 | 0.6522 | 0.6177 | 0.5307 | 0.5353 | **0.7782** |
| 2553 | MG-G2L | 123 | 0.0535 | 0.1678 | 0.3290 | 0.5957 | 0.5787 | 0.5903 | 0.6473 | 0.6768 | 0.8018 | **0.7707** |
| 10000 | Paper 1 | 17 | 0.0095 | 0.0437 | 0.0952 | 0.0688 | 0.1865 | 0.4302 | 0.2975 | 0.3855 | 0.2475 | **0.5760** |
| 10000 | Paper 1 | 42 | 0.0310 | 0.0947 | 0.0125 | 0.0160 | 0.0212 | 0.0280 | 0.0563 | 0.0457 | 0.0370 | **0.0602** |
| 10000 | Paper 1 | 123 | 0.0085 | 0.0860 | 0.6475 | 0.3195 | 0.3790 | 0.4027 | 0.2993 | 0.2205 | 0.5610 | **0.9088** |
| 10000 | MG-G2L | 17 | 0.2417 | 0.6088 | 0.7225 | 0.7632 | 0.7313 | 0.7733 | 0.8557 | 0.8330 | 0.9845 | **0.9970** |
| 10000 | MG-G2L | 42 | 0.2530 | 0.6910 | 0.3632 | 0.7477 | 0.8955 | 0.9882 | 0.9415 | 0.9928 | 0.9443 | **0.9947** |
| 10000 | MG-G2L | 123 | 0.1047 | 0.6990 | 0.8425 | 0.9312 | 0.9760 | 0.9483 | 0.9838 | 0.9198 | 0.9620 | **0.9738** |

## What is retained

- `attempt1_logs/` — the 12 SLURM stdout logs (originals also remain in `logs/`).
- `attempt1_per_epoch/` — 12 CSVs, one per run, with every epoch's losses
  (incl. granularity) and validation R@1 / R@5 / R@10 / MRR. These are the
  citable evidence that Paper 1 is unstable at small scale.
- Checkpoint directories under `saved_models/` are kept as empty shells; only
  the `.pth` files were removed.

Superseded by attempt 2, which matches the full-MIMIC budget of 30,450 gradient
steps at both sizes.
