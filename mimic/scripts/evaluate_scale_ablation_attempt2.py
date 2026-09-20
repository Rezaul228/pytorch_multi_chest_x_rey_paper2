#!/usr/bin/env python3
"""
STEP 4: evaluate the 12 MIMIC scale-ablation attempt-2 checkpoints (jobs
115261-115272) on the SAME MIMIC test set, SAME CheXbert labels, SAME restricted
subset (n = 10,343) and SAME graded protocol as the main 6-seed analysis, so the
numbers are directly comparable to the full-data I->T nDCG@10 delta of +0.0157.

Checkpoints: export/checkpoint_best.pth (best-val policy).
Scoring: mimic/scripts/paper2_graded_relevance_eval build_binary_matrix /
compute_metrics_for_direction, reused UNCHANGED.
MG-G2L is evaluated WITH section args and the test boundary CSV; fallback_count
is asserted 0.

Descriptive only -- no significance tests, no p-values (2-3 pairs per size).

Outputs (NEW files under mimic/results/scale_ablation/):
  graded_relevance_attempt2.csv        per size/arm/seed, every metric, both directions
  retrieval_attempt2.csv               R@1/R@5 per size/arm/seed, both directions
  per_query_attempt2_scale{sz}_seed{seed}.csv   per-query cache (study_id-keyed)
"""
import argparse, csv, hashlib, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
P = paths.repo_root()
R = os.path.join(P, "mimic", "results", "scale_ablation")
sys.path.insert(0, os.path.join(P, "mimic", "scripts"))
import config
config.switch_dataset("mimic_shards_hybrid_full_ori")
import paper2_graded_relevance_eval as scoring  # noqa: E402

LABELS = os.path.join(P, "mimic", "data", "test_labels_chexbert_binary.csv")
SHARD = "mimic_shards_hybrid_full_ori"
BATCH = scoring.BATCH_SIZE
TOP_K = scoring.TOP_K
SEEDS = [17, 42, 123]
SIZES = {2553: 1522, 10000: 385}
ARCH = {"Paper 1": ("v1", 281), "MG-G2L": ("MGG2L_paper2", 289)}
EXP = "mimic_shards_hybrid_full_ori_vo10805_to128_lr1e-4_b128_ep{ep}_dualbr_sy065_main_loss20_ortho15__branch_{br}_scale{sz}_seed_{sd}"
METRICS = [("nDCG@5", "ndcg5"), ("nDCG@10", "ndcg10"), ("mAP@10", "map10"), ("Precision@5", "prec5")]


def ckpt(sz, arm, seed):
    return os.path.join(P, "saved_models", EXP.format(ep=SIZES[sz], br=ARCH[arm][0], sz=sz, sd=seed),
                        "export", "checkpoint_best.pth")


def kwargs():
    c = config.get_current_config()
    return dict(vocab_size=config.get_vocab_size(), embed_dim=config.get_embed_dim(),
                num_heads=c["num_heads"], num_layers=c["num_layers"])


def ids_for_labels(ids):
    idx = pd.read_csv(LABELS, usecols=["study_id"])["study_id"]
    return np.array([int(x) for x in ids]) if pd.api.types.is_integer_dtype(idx) else np.array([str(x) for x in ids], dtype=object)


def encode(sz, arm, seed, device):
    if arm == "Paper 1":
        from base_models_refactored_v1 import MultimodalFusion
        from data_loader_v1 import IndianaDataLoader
    else:
        from base_models_refactored_v1_paper2 import MultimodalFusion
        from data_loader_v1_paper2 import IndianaDataLoader
    m = MultimodalFusion(**kwargs())
    c = torch.load(ckpt(sz, arm, seed), map_location="cpu", weights_only=False)
    r = m.load_state_dict(c["model_state_dict"], strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    assert len(c["model_state_dict"]) == ARCH[arm][1]
    m.eval().to(device)
    dl = IndianaDataLoader(batch_size=BATCH, use_shards=True, shard_subfolder=SHARD)
    dl.tokenizer = scoring.load_tokenizer_from_metadata(SHARD)
    dl.load_data(max_samples=None, skip_processing=True)
    ds = dl.get_test_data(num_samples=None)
    if arm != "Paper 1":
        ds.reset_fallback_count()
    I, T, ids = [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=BATCH, shuffle=False):
            im = b["images"]
            if im.dim() == 4 and im.shape[-1] == 3:
                im = im.permute(0, 3, 1, 2)
            im, cp = im.to(device), b["captions"].to(device)
            if arm == "Paper 1":
                ie, te = m((im, cp), training=False)
            else:
                o = m((im, cp), training=False, findings_token_count=b["findings_token_count"].to(device),
                      has_find=b["has_find"].to(device), has_imp=b["has_imp"].to(device), token_ids=cp)
                ie, te = o[0], o[1]
            I.append(ie.cpu().numpy()); T.append(te.cpu().numpy()); ids += [str(x) for x in b["study_ids"]]
    if arm != "Paper 1":
        print(f"    [MG-G2L] test fallback_count = {ds.fallback_count}/{len(ds)}")
        assert ds.fallback_count == 0, f"fallback_count={ds.fallback_count}"
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(I), np.concatenate(T), np.array(ids, dtype=object)


def score(I, T, ids, device):
    n = len(ids)
    it, tt = torch.FloatTensor(I).to(device), torch.FloatTensor(T).to(device)
    i2t, t2i = torch.matmul(it, tt.T), torch.matmul(tt, it.T)
    B, _ = scoring.build_binary_matrix(ids_for_labels(ids), LABELS)
    Bt = torch.FloatTensor(B).to(device)
    Rm = torch.matmul(Bt, Bt.T).cpu().numpy().astype(np.int8); del Bt
    ideal = (-np.sort(-Rm, axis=1))[:, :TOP_K]
    tot = np.sum(Rm > 0, axis=1)
    restricted = tot > 0
    out, per = {}, {}
    for dn, dk, sim in [("Image->Text", "i2t", i2t), ("Text->Image", "t2i", t2i)]:
        n5, n10, ap, p5 = scoring.compute_metrics_for_direction(sim, Rm, ideal, tot, n)
        per[dk] = dict(ndcg5=n5, ndcg10=n10, map10=ap, prec5=p5)
        for mn, arr in [("nDCG@5", n5), ("nDCG@10", n10), ("mAP@10", ap), ("Precision@5", p5)]:
            out[(mn, dn)] = float(arr[restricted].mean())
        order = np.argsort(-sim.cpu().numpy(), axis=1)[:, :5]
        corr = np.arange(n)[:, None]
        out[("R@1", dn)] = float(np.mean(np.any(order[:, :1] == corr, axis=1)))
        out[("R@5", dn)] = float(np.mean(np.any(order[:, :5] == corr, axis=1)))
    del i2t, t2i
    return out, per, restricted, int(restricted.sum())


def main():
    os.makedirs(R, exist_ok=True)
    for f in ["graded_relevance_attempt2.csv", "retrieval_attempt2.csv"]:
        assert not os.path.exists(os.path.join(R, f)), f"refusing to overwrite {f}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  labels={LABELS}")
    res, cache, rn, ref = {}, {}, None, None
    for sz in SIZES:
        for seed in SEEDS:
            for arm in ARCH:
                print(f"\n=== size {sz}  {arm}  seed {seed}")
                I, T, ids = encode(sz, arm, seed, device)
                if ref is None: ref = ids
                assert np.array_equal(ids, ref)
                o, per, restricted, n = score(I, T, ids, device)
                rn = n; res[(sz, arm, seed)] = o; cache.setdefault((sz, seed), {})[arm] = per
                del I, T
            p1, mg = cache[(sz, seed)]["Paper 1"], cache[(sz, seed)]["MG-G2L"]
            df = pd.DataFrame({"study_id": ref, "restricted": restricted.astype(int)})
            for d in ("i2t", "t2i"):
                for k in ("ndcg5", "ndcg10", "map10", "prec5"):
                    df[f"p1_{k}_{d}"] = p1[d][k]; df[f"mg_{k}_{d}"] = mg[d][k]
            df.to_csv(os.path.join(R, f"per_query_attempt2_scale{sz}_seed{seed}.csv"), index=False)

    with open(os.path.join(R, "graded_relevance_attempt2.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["size", "arm", "seed", "metric", "direction", "restricted_subset_value", "restricted_subset_n"])
        for (sz, arm, seed), o in res.items():
            for (mn, dn), v in o.items():
                if mn.startswith("R@"): continue
                w.writerow([sz, arm, seed, mn, dn, f"{v:.6f}", rn])
    with open(os.path.join(R, "retrieval_attempt2.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["size", "arm", "seed", "direction", "R@1", "R@5"])
        for (sz, arm, seed), o in res.items():
            for dn in ("Image->Text", "Text->Image"):
                w.writerow([sz, arm, seed, dn, f"{o[('R@1', dn)]:.6f}", f"{o[('R@5', dn)]:.6f}"])

    CONV = {2553: [17, 42], 10000: [17, 123]}
    print("\n" + "=" * 110)
    print(f"STEP 4 RESULTS — MIMIC test, CheXbert labels, restricted subset n = {rn}; checkpoint policy = BEST-VAL")
    print("Descriptive only: no significance tests, no p-values.")
    print("=" * 110)
    for sz in SIZES:
        for dn, tag in [("Image->Text", "(a) I->T"), ("Text->Image", "(b) T->I")]:
            print(f"\n--- size {sz}: {tag} per-seed deltas (MG-G2L minus Paper 1) ---")
            print(f"{'metric':<12}|" + "".join(f"{'seed '+str(s):<12}|" for s in SEEDS) + f"{'mean ALL 3':<12}|{'mean CONVERGED':<15}")
            for mn, _ in METRICS:
                ds = {s: res[(sz, 'MG-G2L', s)][(mn, dn)] - res[(sz, 'Paper 1', s)][(mn, dn)] for s in SEEDS}
                allm = np.mean([ds[s] for s in SEEDS]); cvm = np.mean([ds[s] for s in CONV[sz]])
                print(f"{mn:<12}|" + "".join(f"{ds[s]:<+12.4f}|" for s in SEEDS) + f"{allm:<+12.4f}|{cvm:<+15.4f}")
            print(f"  (converged pairs at {sz}: seeds {CONV[sz]})")
        print(f"\n--- size {sz}: (d) R@1 / R@5 per seed per arm ---")
        print(f"{'arm':<8}|{'seed':<5}|{'I->T R@1':<10}|{'I->T R@5':<10}|{'T->I R@1':<10}|{'T->I R@5':<10}")
        for arm in ARCH:
            for s in SEEDS:
                o = res[(sz, arm, s)]
                print(f"{arm:<8}|{s:<5}|{o[('R@1','Image->Text')]:<10.4f}|{o[('R@5','Image->Text')]:<10.4f}|{o[('R@1','Text->Image')]:<10.4f}|{o[('R@5','Text->Image')]:<10.4f}")
        for arm in ARCH:
            v = np.array([res[(sz, arm, s)][('R@1', 'Image->Text')] for s in SEEDS])
            print(f"  {arm:<8} I->T R@1: mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}")
    print("\nsaved: graded_relevance_attempt2.csv, retrieval_attempt2.csv, per_query_attempt2_*.csv")


if __name__ == "__main__":
    main()
