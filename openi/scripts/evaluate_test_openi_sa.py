#!/usr/bin/env python3
"""
Open-I (openi_sa) TEST evaluation, 6 matched seeds, both arms.

Arms
  Paper 1 : ..._branch_v1_seed_{s}_rerun          (fresh reruns, no resume, V100)
  MG-G2L  : ..._branch_MGG2L_paper2_seed_{s}
Both evaluated at their BEST-VAL weights (export/checkpoint_best.pth).

Step 0 pre-flight (hard gate; exits non-zero on any failure)
  - SHA-256 of all 12 checkpoints, pairwise distinct within AND across arms
  - strict=True load, 0 missing / 0 unexpected, tensor counts 281 / 289
  - test split: 752 rows, boundary CSV path + entries quoted, fallback_count == 0
  - MG-G2L eval MUST pass section args: asserted by running the model twice on
    one batch (with and without the args) and requiring the outputs to DIFFER.
    A section-blind eval would repeat the ReXGradient failure at test time.

Step 1 R@K / MRR, both directions, under TWO hit rules
  strict  : hit iff the retrieved row index == the query row index
  dup_text: hit iff the retrieved caption TOKEN SEQUENCE equals the true one
            (shards store no raw text; the 128-int caption IS the text)

Step 2 graded relevance under CheXbert binary labels, restricted subset
  (>=1 positive finding excluding "No Finding"), both directions,
  nDCG@5/@10, mAP@10, P@5 -- scoring reused UNCHANGED from
  mimic/scripts/paper2_graded_relevance_eval.py.

Step 3 per-query caches (both directions) for the CPU statistics step.

Writes NEW files under openi/results/ with openi_sa in the name. MIMIC results
are never read or written.
"""
import argparse
import csv
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
P = paths.repo_root()
R = os.path.join(P, "openi", "results")
sys.path.insert(0, os.path.join(P, "mimic", "scripts"))

import config
config.switch_dataset("openi_sa")                      # MUST precede model/data construction
import paper2_graded_relevance_eval as scoring          # noqa: E402  reused, unchanged

SEEDS = [17, 42, 123, 1337, 2021, 3407]
EXP = "openi_sa_vo10805_to128_lr1e-4_b128_ep100_dualbr_sy065_main_loss20_ortho15__branch_{arch}"
ARCHS = {"Paper1": ("v1_seed_{seed}_rerun", 281), "MG-G2L": ("MGG2L_paper2_seed_{seed}", 289)}
LABELS = os.path.join(P, "openi", "data", "test_labels_chexbert_binary.csv")
BOUNDARIES = os.path.join(P, "openi", "data", "section_boundaries_test_openi_sa.csv")
SHARD_SUBFOLDER = "openi_sa"
BATCH = 64
TOP_K = scoring.TOP_K
K_VALUES = [1, 5, 10]


def ckpt_path(arch, seed):
    return os.path.join(P, "saved_models", EXP.format(arch=ARCHS[arch][0].format(seed=seed)),
                        "export", "checkpoint_best.pth")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_kwargs():
    c = config.get_current_config()
    return dict(vocab_size=config.get_vocab_size(), embed_dim=config.get_embed_dim(),
                num_heads=c["num_heads"], num_layers=c["num_layers"])


def build(arch):
    if arch == "Paper1":
        from base_models_refactored_v1 import MultimodalFusion
    else:
        from base_models_refactored_v1_paper2 import MultimodalFusion
    return MultimodalFusion(**model_kwargs())


def load_into(arch, seed):
    ck = torch.load(ckpt_path(arch, seed), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    m = build(arch)
    res = m.load_state_dict(sd, strict=True)
    return m, sd, ck, res


def test_dataset(arch):
    if arch == "Paper1":
        from data_loader_v1 import IndianaDataLoader
    else:
        from data_loader_v1_paper2 import IndianaDataLoader
    dl = IndianaDataLoader(batch_size=BATCH, use_shards=True, shard_subfolder=SHARD_SUBFOLDER)
    dl.tokenizer = scoring.load_tokenizer_from_metadata(SHARD_SUBFOLDER)
    dl.load_data(max_samples=None, skip_processing=True)
    return dl.get_test_data(num_samples=None)


# --------------------------------------------------------------------- step 0
def preflight():
    print("=" * 100 + "\nSTEP 0: PRE-FLIGHT\n" + "=" * 100)
    fails, hashes, rows = [], {}, []
    for arch in ARCHS:
        for seed in SEEDS:
            p = ckpt_path(arch, seed)
            if not os.path.exists(p):
                fails.append(f"{arch} seed {seed}: MISSING {p}"); continue
            h = sha256(p)
            if h in hashes:
                fails.append(f"DUPLICATE checkpoint: {arch}/{seed} == {hashes[h]}")
            hashes[h] = f"{arch}/{seed}"
            m, sd, ck, res = load_into(arch, seed)
            n_t = len(sd)
            ok = (not res.missing_keys) and (not res.unexpected_keys)
            if not ok:
                fails.append(f"{arch} seed {seed}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
            if n_t != ARCHS[arch][1]:
                fails.append(f"{arch} seed {seed}: {n_t} tensors != {ARCHS[arch][1]}")
            rows.append({"arch": arch, "seed": seed, "file": "checkpoint_best.pth", "sha12": h[:12],
                         "n_tensors": n_t, "strict_load": "OK" if ok else "MISMATCH",
                         "best_epoch": ck.get("epoch"), "best_val_r1_avg": ck.get("val_r1_avg")})
            del m
    print(f"\n| arch | seed | evaluated file | sha256[:12] | tensors | strict-load | best epoch | best val R@1 avg |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        bv = r["best_val_r1_avg"]
        print(f"| {r['arch']} | {r['seed']} | {r['file']} | {r['sha12']} | {r['n_tensors']} | {r['strict_load']} "
              f"| {r['best_epoch']} | {bv if bv is None else f'{bv:.4f}'} |")
    print(f"\ncheckpoints={len(rows)}/12  distinct={'yes' if len(hashes) == len(rows) else 'NO'}")

    ds = test_dataset("MG-G2L")
    print(f"test rows={len(ds)}  boundary CSV={ds.section_boundaries_path}  entries={len(ds.section_boundaries)}")
    ds.reset_fallback_count()
    for i in range(len(ds)):
        _ = ds[i]
    print(f"fallback_count after touching all {len(ds)} test samples = {ds.fallback_count}")
    if len(ds) != 752:
        fails.append(f"test rows {len(ds)} != 752")
    if ds.fallback_count != 0:
        fails.append(f"test fallback_count={ds.fallback_count} != 0")

    # section args really reach the model at EVAL time
    m, _, _, _ = load_into("MG-G2L", SEEDS[0])
    m.eval()
    b = next(iter(DataLoader(ds, batch_size=8, shuffle=False)))
    im = b["images"].permute(0, 3, 1, 2) if b["images"].shape[-1] == 3 else b["images"]
    with torch.no_grad():
        with_args = m((im, b["captions"]), training=False,
                      findings_token_count=b["findings_token_count"], has_find=b["has_find"],
                      has_imp=b["has_imp"], token_ids=b["captions"])
        no_args = m((im, b["captions"]), training=False)
    differ = not torch.equal(with_args[1], no_args[1])
    print(f"section args change the MG-G2L output on a real test batch: {differ}  "
          f"(max|diff| text_emb = {(with_args[1] - no_args[1]).abs().max().item():.3e}; "
          f"has_find={int(b['has_find'].sum())}/8 has_imp={int(b['has_imp'].sum())}/8)")
    if not differ:
        fails.append("section args do NOT change MG-G2L output -> evaluation would be section-blind")
    del m

    print("\nPRE-FLIGHT=" + ("PASS" if not fails else "FAIL"))
    for f in fails:
        print("  - " + f)
    if fails:
        sys.exit(1)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- encode
def encode(arch, seed, device):
    ds = test_dataset(arch)
    if arch != "Paper1":
        ds.reset_fallback_count()
    m, _, _, _ = load_into(arch, seed)
    m.eval().to(device)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False)
    I, T, ids, caps = [], [], [], []
    with torch.no_grad():
        for b in loader:
            im = b["images"]
            if im.dim() == 4 and im.shape[-1] == 3:
                im = im.permute(0, 3, 1, 2)
            im, cp = im.to(device), b["captions"].to(device)
            if arch == "Paper1":
                ie, te = m((im, cp), training=False)
            else:
                o = m((im, cp), training=False,
                      findings_token_count=b["findings_token_count"].to(device),
                      has_find=b["has_find"].to(device), has_imp=b["has_imp"].to(device), token_ids=cp)
                ie, te = o[0], o[1]
            I.append(ie.cpu().numpy()); T.append(te.cpu().numpy())
            ids += [str(x) for x in b["study_ids"]]
            caps.append(b["captions"].numpy())
    fb = getattr(ds, "fallback_count", None)
    if arch != "Paper1":
        print(f"    [MG-G2L] fallback_count during encoding: {fb}/{len(ds)}")
        assert fb == 0, f"MG-G2L test fallback_count={fb} (expected 0)"
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(I), np.concatenate(T), np.array(ids, dtype=object), np.concatenate(caps)


# --------------------------------------------------------------------- step 1
def retrieval(sim, gid):
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)[:, :max(K_VALUES)]
    out = {}
    # strict row-id rule (same formula as evaluate_cross_modal_retrieval_streaming)
    ranks = np.empty(n)
    for i in range(n):
        s, c = sim[i], sim[i, i]
        r = 1 + np.sum(s > c)
        ties = np.sum(np.isclose(s, c)) - 1
        ranks[i] = r + ties / 2 if ties > 0 else r
    m = {"mrr": float(np.mean(1 / ranks)), "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks))}
    for k in K_VALUES:
        m[f"recall@{k}"] = float(np.mean(np.any(order[:, :k] == np.arange(n)[:, None], axis=1)))
    out["strict"] = m
    # duplicate-caption-text rule
    same = gid[:, None] == gid[None, :]
    bp = np.where(same, sim, -np.inf).max(axis=1)
    rd = 1 + np.sum(sim > bp[:, None], axis=1)
    m = {"mrr": float(np.mean(1 / rd)), "mean_rank": float(rd.mean()), "median_rank": float(np.median(rd))}
    for k in K_VALUES:
        m[f"recall@{k}"] = float(np.mean(np.any(same[np.arange(n)[:, None], order[:, :k]], axis=1)))
    out["dup_text"] = m
    return out


# --------------------------------------------------------------------- step 2
def graded(I, T, ids, device):
    n = len(ids)
    it, tt = torch.FloatTensor(I).to(device), torch.FloatTensor(T).to(device)
    i2t, t2i = torch.matmul(it, tt.T), torch.matmul(tt, it.T)
    B, cols = scoring.build_binary_matrix(ids, LABELS)
    Bt = torch.FloatTensor(B).to(device)
    Rm = torch.matmul(Bt, Bt.T).cpu().numpy().astype(np.int8)
    del Bt
    ideal = (-np.sort(-Rm, axis=1))[:, :TOP_K]
    tot = np.sum(Rm > 0, axis=1)
    restricted = tot > 0
    rows, per = [], {}
    for dname, dk, sim in [("Image->Text", "i2t", i2t), ("Text->Image", "t2i", t2i)]:
        n5, n10, ap, p5 = scoring.compute_metrics_for_direction(sim, Rm, ideal, tot, n)
        per[dk] = {"ndcg5": n5, "ndcg10": n10, "map10": ap, "prec5": p5}
        for mn, arr in [("nDCG@5", n5), ("nDCG@10", n10), ("mAP@10", ap), ("Precision@5", p5)]:
            rows.append({"metric": mn, "direction": dname,
                         "full_corpus_value": round(float(arr.mean()), 4), "full_corpus_n": n,
                         "restricted_subset_value": round(float(arr[restricted].mean()), 4),
                         "restricted_subset_n": int(restricted.sum())})
    return rows, per, restricted, i2t.cpu().numpy(), t2i.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    for f in ["retrieval_metrics_openi_sa.csv", "paper1_baseline_graded_relevance_openi_sa.csv",
              "mg_g2l_graded_relevance_openi_sa.csv", "duplicate_caption_groups_openi_sa.csv",
              "preflight_openi_sa.csv"]:
        assert not os.path.exists(os.path.join(R, f)), f"refusing to overwrite {f}"

    pf = preflight()
    pf.to_csv(os.path.join(R, "preflight_openi_sa.csv"), index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 100 + f"\nSTEPS 1-3: EVALUATION on {device}\n" + "=" * 100)
    ret, gr = [], {"Paper1": {}, "MG-G2L": {}}
    ref_ids = dup = None
    cache = {}
    for seed in a.seeds:
        for arch in ARCHS:
            print(f"\n=== {arch} seed {seed} -> {ckpt_path(arch, seed)}")
            I, T, ids, caps = encode(arch, seed, device)
            if ref_ids is None:
                ref_ids = ids
                keys = np.array([hashlib.sha1(c.astype(np.int64).tobytes()).hexdigest() for c in caps], dtype=object)
                vc = pd.Series(keys).value_counts(); g = vc[vc >= 2]
                dup = {"n_test": len(ids), "n_unique_captions": int(len(vc)), "n_duplicate_groups": int(len(g)),
                       "n_rows_in_duplicate_groups": int(g.sum()), "largest_group": int(vc.max())}
                gid = pd.factorize(pd.Series(keys))[0]
                print(f"  duplicate-caption groups: {dup}")
            assert np.array_equal(ids, ref_ids), "study_id order differs across runs"
            g_rows, per, restricted, i2t, t2i = graded(I, T, ids, device)
            gr[arch][seed] = g_rows
            for dname, sim in [("i2t", i2t), ("t2i", t2i)]:
                for rule, mm in retrieval(sim, gid).items():
                    ret.append({"architecture": arch, "seed": seed, "direction": dname, "rule": rule,
                                "n_test": len(ids), **mm})
            cache.setdefault(seed, {})[arch] = (per, restricted)
            del I, T, i2t, t2i

        p1, mg = cache[seed]["Paper1"], cache[seed]["MG-G2L"]
        df = pd.DataFrame({"study_id": ref_ids, "restricted": p1[1].astype(int)})
        for d in ("i2t", "t2i"):
            for k in ("ndcg5", "ndcg10", "map10", "prec5"):
                df[f"p1_{k}_{d}"] = p1[0][d][k]; df[f"mg_{k}_{d}"] = mg[0][d][k]
        pq = os.path.join(R, f"per_query_metrics_openi_sa_seed_{seed}.csv")
        df.to_csv(pq, index=False); print(f"  saved {pq} ({len(df)} rows)")

    pd.DataFrame(ret).to_csv(os.path.join(R, "retrieval_metrics_openi_sa.csv"), index=False)
    pd.DataFrame([dup]).to_csv(os.path.join(R, "duplicate_caption_groups_openi_sa.csv"), index=False)
    for arch, fn in [("Paper1", "paper1_baseline_graded_relevance_openi_sa.csv"),
                     ("MG-G2L", "mg_g2l_graded_relevance_openi_sa.csv")]:
        with open(os.path.join(R, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "metric", "direction", "full_corpus_value", "full_corpus_n",
                                              "restricted_subset_value", "restricted_subset_n"])
            w.writeheader()
            for s in a.seeds:
                for r in gr[arch][s]:
                    w.writerow({"seed": s, **r})

    rn = gr["Paper1"][a.seeds[0]][0]["restricted_subset_n"]
    print("\n===== FINAL SUMMARY (OPEN-I TEST EVALUATION) =====")
    print(f"test_n={dup['n_test']}  restricted_n(CheXbert)={rn}  seeds={a.seeds}")
    print(f"duplicate_caption_groups={dup['n_duplicate_groups']}  rows_in_groups={dup['n_rows_in_duplicate_groups']}  "
          f"unique_captions={dup['n_unique_captions']}  largest_group={dup['largest_group']}")
    rd = pd.DataFrame(ret)
    for rule in ["dup_text", "strict"]:
        print(f"\n-- R@K / MRR, rule={rule} --")
        print(f"{'arch':<7} {'seed':<5} {'dir':<4} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7}")
        for arch in ARCHS:
            for d in ("i2t", "t2i"):
                sub = rd[(rd.architecture == arch) & (rd.direction == d) & (rd.rule == rule)].sort_values("seed")
                for _, r in sub.iterrows():
                    print(f"{arch:<7} {int(r.seed):<5} {d:<4} {r['recall@1']:7.4f} {r['recall@5']:7.4f} {r['recall@10']:7.4f} {r.mrr:7.4f}")
                f = lambda c: f"{sub[c].mean():.4f}+-{sub[c].std(ddof=1):.4f}"
                print(f"{arch:<7} {'mean':<5} {d:<4} {f('recall@1'):>16} {f('recall@5'):>16} {f('recall@10'):>16} {f('mrr'):>16}")
    print(f"\n-- graded relevance, RESTRICTED n={rn} (mean +- sd over seeds, ddof=1) --")
    print(f"{'metric':<12} {'direction':<12} {'Paper1':<17} {'MG-G2L':<17} {'delta':>9} {'MG>P1':>7}")
    for i, r0 in enumerate(gr["Paper1"][a.seeds[0]]):
        p1v = np.array([gr["Paper1"][s][i]["restricted_subset_value"] for s in a.seeds])
        mgv = np.array([gr["MG-G2L"][s][i]["restricted_subset_value"] for s in a.seeds])
        print(f"{r0['metric']:<12} {r0['direction']:<12} {p1v.mean():.4f}+-{p1v.std(ddof=1):.4f}   "
              f"{mgv.mean():.4f}+-{mgv.std(ddof=1):.4f}   {mgv.mean()-p1v.mean():+9.4f} {int((mgv>p1v).sum()):>4}/{len(a.seeds)}")
    print("===== END SUMMARY =====")


if __name__ == "__main__":
    main()
