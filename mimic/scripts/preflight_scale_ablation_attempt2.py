#!/usr/bin/env python3
"""
STEP 2 pre-flight for the MIMIC scale-ablation attempt-2 checkpoints (jobs
115261-115272). Report only; exits non-zero on any failure.

  1. SHA-256 of all 12 checkpoint_best.pth -> pairwise distinct.
  2. strict=True load, 0 missing / 0 unexpected; 281 tensors Paper 1, 289 MG-G2L.
  3. Section gates in the 6 MG-G2L checkpoints vs a fresh init under the same
     seed: the 4 synergy_branch gates must have MOVED, the 4 difference_branch
     gates must be BIT-IDENTICAL to init.
     Architectural reason: MultimodalFusion.forward calls
       self.synergy_branch(..., findings_token_count=..., has_find=..., ...)
     but
       self.difference_branch(image_tokens, text_tokens)
     with no section args, so the difference branch's section_aware_feedback is
     never invoked, its gates never enter the autograd graph, .grad stays None
     and the optimiser skips them (weight decay included).
"""
import hashlib, os, random, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
P = paths.repo_root()
import config
config.switch_dataset("mimic_shards_hybrid_full_ori")

RUNS = [(2553, "Paper 1", "v1", 1522, 281), (2553, "MG-G2L", "MGG2L_paper2", 1522, 289),
        (10000, "Paper 1", "v1", 385, 281), (10000, "MG-G2L", "MGG2L_paper2", 385, 289)]
SEEDS = [17, 42, 123]
EXP = "mimic_shards_hybrid_full_ori_vo10805_to128_lr1e-4_b128_ep{ep}_dualbr_sy065_main_loss20_ortho15__branch_{br}_scale{sz}_seed_{sd}"


def ck(sz, br, ep, sd):
    return os.path.join(P, "saved_models", EXP.format(ep=ep, br=br, sz=sz, sd=sd), "export", "checkpoint_best.pth")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kw():
    c = config.get_current_config()
    return dict(vocab_size=config.get_vocab_size(), embed_dim=config.get_embed_dim(),
                num_heads=c["num_heads"], num_layers=c["num_layers"])


def build(arch, seed=None):
    if seed is not None:
        torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        np.random.seed(seed); random.seed(seed)
    if arch == "Paper 1":
        from base_models_refactored_v1 import MultimodalFusion
    else:
        from base_models_refactored_v1_paper2 import MultimodalFusion
    return MultimodalFusion(**kw())


def main():
    fails, hashes, rows = [], {}, []
    print("2.1 / 2.2  CHECKSUM + STRICT LOAD\n")
    print(f"{'size':<6}|{'arm':<8}|{'seed':<5}|{'sha256[:12]':<13}|{'tensors':<8}|{'strict-load':<12}|{'best_ep':<8}")
    print("-" * 70)
    for sz, lbl, br, ep, ntens in RUNS:
        for sd in SEEDS:
            p = ck(sz, br, ep, sd)
            if not os.path.exists(p):
                fails.append(f"MISSING {p}"); continue
            h = sha(p)
            if h in hashes:
                fails.append(f"DUPLICATE: {lbl}/{sz}/{sd} == {hashes[h]}")
            hashes[h] = f"{lbl}/{sz}/{sd}"
            c = torch.load(p, map_location="cpu", weights_only=False)
            sd_ = c["model_state_dict"]
            m = build(lbl)
            r = m.load_state_dict(sd_, strict=True)
            ok = not r.missing_keys and not r.unexpected_keys
            if not ok: fails.append(f"{lbl}/{sz}/{sd}: missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}")
            if len(sd_) != ntens: fails.append(f"{lbl}/{sz}/{sd}: {len(sd_)} tensors != {ntens}")
            print(f"{sz:<6}|{lbl:<8}|{sd:<5}|{h[:12]:<13}|{len(sd_):<8}|{'OK' if ok else 'MISMATCH':<12}|{c.get('epoch'):<8}")
            rows.append((sz, lbl, sd, sd_))
    print(f"\n12 checkpoints, {len(hashes)} distinct SHA-256 -> {'all distinct' if len(hashes)==12 else 'DUPLICATES PRESENT'}")

    print("\n\n2.3  SECTION GATES vs FRESH INIT (MG-G2L only)\n")
    print(f"{'size':<6}|{'seed':<5}|{'branch':<11}|{'gate':<10}|{'layer':<6}|{'equal_to_init':<14}|{'max|diff|':<11}")
    print("-" * 76)
    summary = {}
    for sz, lbl, sd, state in rows:
        if lbl != "MG-G2L":
            continue
        fresh = build("MG-G2L", seed=sd).state_dict()
        nsyn_moved = ndiff_same = 0
        for br in ("synergy_branch", "difference_branch"):
            for layer in (0, 1):
                for gate in ("section_find_gate_weights", "section_imp_gate_weights"):
                    k = f"{br}.co_attn_layers.{layer}.{gate}"
                    eq = torch.equal(state[k], fresh[k])
                    d = (state[k] - fresh[k]).abs().max().item()
                    if br == "synergy_branch" and not eq: nsyn_moved += 1
                    if br == "difference_branch" and eq: ndiff_same += 1
                    print(f"{sz:<6}|{sd:<5}|{br:<11}|{gate.replace('section_','').replace('_gate_weights',''):<10}|{layer:<6}|{str(eq):<14}|{d:<11.4e}")
        summary[(sz, sd)] = (nsyn_moved, ndiff_same)
        if nsyn_moved != 4: fails.append(f"MG-G2L {sz}/{sd}: {nsyn_moved}/4 synergy gates moved from init (expected 4)")
        if ndiff_same != 4: fails.append(f"MG-G2L {sz}/{sd}: {ndiff_same}/4 difference gates identical to init (expected 4)")
    print(f"\n{'size':<6}|{'seed':<5}|{'synergy gates MOVED':<21}|{'difference gates IDENTICAL':<27}")
    for (sz, sd), (a, b) in summary.items():
        print(f"{sz:<6}|{sd:<5}|{f'{a}/4':<21}|{f'{b}/4':<27}")

    print("\n===== STEP 2 =====")
    print("PREFLIGHT=" + ("PASS" if not fails else "FAIL"))
    for f in fails: print("  - " + f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
