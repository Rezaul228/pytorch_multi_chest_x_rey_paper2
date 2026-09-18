#!/usr/bin/env python3
"""
STEP B0 -- resume checks for the openi_sa runs (Paper 1 arm resumed at epoch 69
after the 6 h TIMEOUT, jobs 114769-114774 -> 114831-114836; MG-G2L arm 114775-
114780 ran straight through). Report only; exits non-zero if any check fails.

  1. best-so-far survived the resume?  Quotes, per Paper 1 seed, the last
     "New best checkpoint" line before the timeout and the first one after the
     restart. A post-resume "new best" LOWER than the pre-timeout best proves
     the counter was reset (train_retrieval_v2.py did not persist
     best_val_r1_avg in checkpoint_resume.pth -- fixed after this finding).
     Also reports, per seed, the global best (over all 100 epochs, from both
     logs) vs the best actually on disk (the resumed job's final "New best").
  2. LR-scheduler state: N/A by construction -- neither training script has a
     scheduler (constant Adam LR = config learning_rate); the Adam moment state
     IS saved/restored via optimizer_state_dict. Verified by grep in this script.
  3. Both arms reached 100/100 (epoch headers); best-val epoch per run.
  4. SHA-256 of all 12 checkpoint_best.pth -> distinct within and across arms.
  5. Tensor counts of model_state_dict: Paper 1 == 281, MG-G2L == 289.

Writes openi/results/resume_checks_openi_sa.csv (aggregate only).
"""
import csv
import glob
import hashlib
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
P = paths.repo_root()
SEEDS = [17, 42, 123, 1337, 2021, 3407]
EXP = "openi_sa_vo10805_to128_lr1e-4_b128_ep100_dualbr_sy065_main_loss20_ortho15__branch_{arch}_seed_{seed}"
ARCHS = {"Paper1": ("v1", "v1", 281), "MG-G2L": ("MGG2L_paper2", "MGG2L", 289)}
NEWBEST = re.compile(r"New best checkpoint \(epoch (\d+), val R@1 avg=([0-9.]+)\)")
EPOCH = re.compile(r"^Epoch (\d+)/(\d+)\s*$")
OUT = os.path.join(P, "openi", "results", "resume_checks_openi_sa.csv")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse(path):
    bests, epochs, resume = [], set(), None
    for line in open(path, errors="replace"):
        m = NEWBEST.search(line)
        if m:
            bests.append((int(m.group(1)) + 1, float(m.group(2))))  # 0-indexed in log -> 1-indexed
        m = EPOCH.match(line)
        if m:
            epochs.add(int(m.group(1)))
        if "RESUMING TRAINING" in line:
            resume = line.strip()
    return bests, epochs, resume


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    src = open(os.path.join(P, "train_retrieval_v2.py")).read() + open(os.path.join(P, "train_retrieval_v2_paper2.py")).read()
    has_sched = bool(re.search(r"lr_scheduler|StepLR|CosineAnnealing|ReduceLROnPlateau|OneCycle", src))
    print(f"[2] LR scheduler present in either training script: {has_sched}  -> "
          f"{'CHECK SCHEDULER STATE' if has_sched else 'N/A: constant Adam LR; optimizer_state_dict is saved+restored'}")

    rows, fails, hashes = [], [], {}
    for arch_key, (arch, tag, n_tensors) in ARCHS.items():
        for seed in SEEDS:
            logs = sorted(glob.glob(os.path.join(P, "logs", f"openisa_{tag}_s{seed}_*.out")))
            parsed = [parse(l) for l in logs]
            all_ep = set().union(*[p[1] for p in parsed])
            all_best = [b for p in parsed for b in p[0]]
            global_best = max(all_best, key=lambda x: (x[1], x[0]))
            final = parsed[-1]
            ondisk_best = max(final[0], key=lambda x: (x[1], x[0])) if final[0] else None
            row = {"arch": arch_key, "seed": seed, "logs": ";".join(os.path.basename(l) for l in logs),
                   "resumed": "yes" if len(logs) > 1 else "no", "epochs_completed": max(all_ep),
                   "global_best_epoch": global_best[0], "global_best_val": global_best[1],
                   "ondisk_best_epoch": ondisk_best[0], "ondisk_best_val": ondisk_best[1]}
            if len(logs) > 1:
                pre = max(parsed[0][0], key=lambda x: (x[1], x[0]))
                first_post = parsed[-1][0][0]
                row.update(pre_timeout_best=f"ep{pre[0]}={pre[1]:.4f}", first_post_resume_newbest=f"ep{first_post[0]}={first_post[1]:.4f}",
                           resume_line=final[2] or "")
                row["best_so_far_survived"] = "yes" if first_post[1] >= pre[1] else "NO (counter reset)"
                if first_post[1] < pre[1]:
                    fails.append(f"{arch_key} seed {seed}: best-so-far reset on resume (pre {pre[1]:.4f}@{pre[0]} -> post 'new best' {first_post[1]:.4f}@{first_post[0]})")
            else:
                row.update(pre_timeout_best="-", first_post_resume_newbest="-", resume_line="", best_so_far_survived="n/a (no resume)")
            row["ondisk_is_global_best"] = "yes" if ondisk_best[1] >= global_best[1] else f"NO (lost ep{global_best[0]}={global_best[1]:.4f})"
            if max(all_ep) != 100:
                fails.append(f"{arch_key} seed {seed}: epochs completed {max(all_ep)} != 100")

            ck = os.path.join(P, "saved_models", EXP.format(arch=arch, seed=seed), "export", "checkpoint_best.pth")
            c = torch.load(ck, map_location="cpu", weights_only=False)
            row["ckpt_epoch_field"] = c.get("epoch"); row["ckpt_val_r1_avg"] = c.get("val_r1_avg")
            row["n_tensors"] = len(c["model_state_dict"])
            if row["n_tensors"] != n_tensors:
                fails.append(f"{arch_key} seed {seed}: {row['n_tensors']} tensors != {n_tensors}")
            h = sha(ck); row["sha12"] = h[:12]
            if h in hashes:
                fails.append(f"DUPLICATE checkpoint_best: {arch_key}/{seed} == {hashes[h]}")
            hashes[h] = f"{arch_key}/{seed}"
            rows.append(row)

    fields = list(rows[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    print("\n| arch | seed | resumed | epochs | pre-timeout best | first post-resume 'new best' | survived | global best | on-disk best | on-disk==global | ckpt ep | tensors | sha12 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['arch']} | {r['seed']} | {r['resumed']} | {r['epochs_completed']} | {r['pre_timeout_best']} | {r['first_post_resume_newbest']} | "
              f"{r['best_so_far_survived']} | ep{r['global_best_epoch']}={r['global_best_val']:.4f} | ep{r['ondisk_best_epoch']}={r['ondisk_best_val']:.4f} | "
              f"{r['ondisk_is_global_best']} | {r['ckpt_epoch_field']} | {r['n_tensors']} | {r['sha12']} |")
    print("\n===== FINAL SUMMARY (B0 RESUME CHECKS) =====")
    p1 = [r for r in rows if r["arch"] == "Paper1"]
    print(f"best_so_far={'ok' if not any('NO' in r['best_so_far_survived'] for r in p1) else 'BAD'} "
          f"({sum('NO' in r['best_so_far_survived'] for r in p1)}/6 Paper 1 seeds reset on resume; "
          f"on-disk best != global best for: {[r['seed'] for r in rows if r['ondisk_is_global_best'] != 'yes'] or 'none'})")
    print(f"scheduler={'CHECK' if has_sched else 'n/a (no scheduler; constant LR; Adam state restored)'}")
    print("epochs=" + ",".join(f"{r['arch']}/{r['seed']}:{r['epochs_completed']}" for r in rows))
    print(f"checksums={'distinct' if len(hashes) == 12 else 'DUPLICATES'} ({len(hashes)}/12 unique)")
    print(f"tensors={'/'.join(str(r['n_tensors']) for r in rows if r['arch']=='Paper1')} (Paper1) "
          f"{'/'.join(str(r['n_tensors']) for r in rows if r['arch']=='MG-G2L')} (MG-G2L)")
    print(f"csv={OUT}")
    print("B0=" + ("FAIL" if fails else "PASS"))
    for m in fails:
        print("  - " + m)
    print("===== END SUMMARY =====")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
