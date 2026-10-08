#!/usr/bin/env python3
"""
Four-part job on the 12 full-MIMIC final-epoch checkpoints. INFERENCE ONLY.

PART 1  cache rankings: per query, both directions, the top-10 retrieved
        study_ids + cosine similarities + the rank of the paired item.
        One file per checkpoint under mimic/results/rankings/.
        SHA-256 of every checkpoint is verified against the committed
        mimic/results/checksum_report_seeds_2021_1337_paper2.json before loading.
PART 2  qualitative examples, seed 3407, I->T, CheXbert restricted set.
PART 3  conditioned on both arms ranking the paired item first: per-seed and
        seed-level deltas, both directions, 4 metrics.
PART 4  oracle upper bound: rank by r_ij itself, both directions, CheXbert and
        CheXpert (label-only, no model).

Scoring reuses mimic/scripts/paper2_graded_relevance_eval unchanged.
Ranking files are study_id-keyed -> gitignored (PhysioNet DUA), see reply.
"""
import csv, hashlib, json, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import ttest_1samp, t as tdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
P = paths.repo_root()
R = os.path.join(P, "mimic", "results")
RK = os.path.join(R, "rankings")
sys.path.insert(0, os.path.join(P, "mimic", "scripts"))
import config
config.switch_dataset("mimic_shards_hybrid_full_ori")
import paper2_graded_relevance_eval as scoring

SEEDS = [17, 42, 123, 1337, 2021, 3407]
SHARD = "mimic_shards_hybrid_full_ori"
CHEXBERT = os.path.join(P, "mimic", "data", "test_labels_chexbert_binary.csv")
CHEXPERT = os.path.join(P, "mimic", "data", "test_labels_chexpert_binary.csv")
CKSUM = os.path.join(R, "checksum_report_seeds_2021_1337_paper2.json")
BASE = "mimic_shards_hybrid_full_orl_vo10805_to128_lr5e-5_b256_ep50_dualbr_sy065_main_loss20_ortho15__branch_"
V1DIR = {s: BASE + f"v1_seed_{s}" + ("_rerun" if s == 2021 else "") for s in SEEDS}
MGDIR = {s: BASE + f"MGG2L_paper2_seed_{s}" for s in SEEDS}
MET = [("nDCG@5", "ndcg5"), ("nDCG@10", "ndcg10"), ("mAP@10", "map10"), ("Precision@5", "prec5")]
TOP = 10


def wpath(arm, seed):
    d = V1DIR[seed] if arm == "Paper 1" else MGDIR[seed]
    return os.path.join(P, "saved_models", d, "export", "model_weights.pth")


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


def encode(arm, seed, device):
    if arm == "Paper 1":
        from base_models_refactored_v1 import MultimodalFusion
        from data_loader_v1 import IndianaDataLoader
    else:
        from base_models_refactored_v1_paper2 import MultimodalFusion
        from data_loader_v1_paper2 import IndianaDataLoader
    m = MultimodalFusion(**kw())
    sd = torch.load(wpath(arm, seed), map_location="cpu", weights_only=True)
    r = m.load_state_dict(sd, strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    m.eval().to(device)
    dl = IndianaDataLoader(batch_size=scoring.BATCH_SIZE, use_shards=True, shard_subfolder=SHARD)
    dl.tokenizer = scoring.load_tokenizer_from_metadata(SHARD)
    dl.load_data(max_samples=None, skip_processing=True)
    ds = dl.get_test_data(num_samples=None)
    if arm != "Paper 1":
        ds.reset_fallback_count()
    I, T, ids = [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=scoring.BATCH_SIZE, shuffle=False):
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
        assert ds.fallback_count == 0, ds.fallback_count
    del m
    torch.cuda.empty_cache()
    return np.concatenate(I), np.concatenate(T), np.array(ids, dtype=object)


def relevance(ids, labels_path):
    idx = pd.read_csv(labels_path, usecols=["study_id"])["study_id"]
    key = np.array([int(x) for x in ids]) if pd.api.types.is_integer_dtype(idx) else np.array([str(x) for x in ids], dtype=object)
    B, cols = scoring.build_binary_matrix(key, labels_path)
    Rm = (B @ B.T).astype(np.int16)
    return Rm, cols


def rank_and_metrics(sim_t, Rm, ideal, tot, n):
    """top-10 ids/sims, paired rank, and the 4 per-query metrics for one direction."""
    sims, idxs = torch.topk(sim_t, k=TOP, dim=1)
    diag = sim_t.diagonal()
    paired_rank = (1 + (sim_t > diag.unsqueeze(1)).sum(dim=1)).cpu().numpy()
    n5, n10, ap, p5 = scoring.compute_metrics_for_direction(sim_t, Rm, ideal, tot, n)
    return idxs.cpu().numpy(), sims.cpu().numpy(), paired_rank, dict(ndcg5=n5, ndcg10=n10, map10=ap, prec5=p5)


def main():
    os.makedirs(RK, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = json.load(open(CKSUM))
    print(f"device={device}\n")

    print("=" * 110); print("PART 1: CACHE RANKINGS (SHA-256 verified against the committed checksum report)"); print("=" * 110)
    print(f"{'arm':<8}|{'seed':<5}|{'sha256[:12]':<13}|{'matches report':<15}|{'file':<42}")
    store = {}
    ids_ref = None
    Rm_cb = ideal_cb = tot_cb = restricted = None
    for arm, dmap, key in (("Paper 1", V1DIR, "v1_checksums"), ("MG-G2L", MGDIR, "mgg2l_checksums")):
        for seed in SEEDS:
            p = wpath(arm, seed)
            h = sha(p)
            exp = ck[key][str(seed)]
            ok = (h == exp)
            assert ok, f"SHA MISMATCH {arm} seed {seed}: got {h[:12]} expected {exp[:12]}"
            I, T, ids = encode(arm, seed, device)
            if ids_ref is None:
                ids_ref = ids
                Rm_cb, cols_cb = relevance(ids, CHEXBERT)
                ideal_cb = (-np.sort(-Rm_cb, axis=1))[:, :TOP]
                tot_cb = (Rm_cb > 0).sum(axis=1)
                restricted = tot_cb > 0
                print(f"\n  test n={len(ids)}  CheXbert restricted n={int(restricted.sum())}  findings={len(cols_cb)}\n")
            assert np.array_equal(ids, ids_ref)
            n = len(ids)
            it = torch.from_numpy(I).to(device); tt = torch.from_numpy(T).to(device)
            rows = []
            res = {}
            for dname, sim in (("i2t", torch.matmul(it, tt.T)), ("t2i", torch.matmul(tt, it.T))):
                idxs, sims, pr, per = rank_and_metrics(sim, Rm_cb, ideal_cb, tot_cb, n)
                res[dname] = dict(idxs=idxs, sims=sims, paired_rank=pr, per=per)
                for i in range(n):
                    rows.append([ids[i], dname, int(pr[i])] +
                                [ids[j] for j in idxs[i]] + [f"{v:.6f}" for v in sims[i]])
                del sim
            del it, tt
            torch.cuda.empty_cache()
            fn = os.path.join(RK, f"rankings_{'v1' if arm=='Paper 1' else 'MGG2L'}_seed{seed}.csv")
            with open(fn, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["study_id", "direction", "paired_rank"] +
                           [f"top{k+1}_study_id" for k in range(TOP)] + [f"top{k+1}_sim" for k in range(TOP)])
                w.writerows(rows)
            store[(arm, seed)] = res
            print(f"{arm:<8}|{seed:<5}|{h[:12]:<13}|{str(ok):<15}|{os.path.basename(fn):<42}")

    # ---------------- PART 3 ----------------
    print("\n" + "=" * 110); print("PART 3: CONDITIONED ON BOTH ARMS RANKING THE PAIRED ITEM FIRST"); print("=" * 110)
    part3 = []
    for dname, dlab in (("i2t", "Image->Text"), ("t2i", "Text->Image")):
        print(f"\n--- {dlab} ---")
        print(f"{'seed':<6}|{'subset n':<9}|" + "".join(f"{m:<13}|" for m, _ in MET))
        per_seed = {}
        for seed in SEEDS:
            a = store[("Paper 1", seed)][dname]; b = store[("MG-G2L", seed)][dname]
            keep = restricted & (a["paired_rank"] == 1) & (b["paired_rank"] == 1)
            d = {mk: float((b["per"][mk][keep] - a["per"][mk][keep]).mean()) for _, mk in MET}
            per_seed[seed] = (int(keep.sum()), d)
            print(f"{seed:<6}|{int(keep.sum()):<9}|" + "".join(f"{d[mk]:<+13.4f}|" for _, mk in MET))
        print(f"{'seed-level':<16}" + "  ".join(f"{m}" for m, _ in MET))
        for mname, mk in MET:
            v = np.array([per_seed[s][1][mk] for s in SEEDS]); nn = len(v); sd = v.std(ddof=1)
            pv = float(ttest_1samp(v, 0.0).pvalue)
            half = tdist.ppf(0.975, nn - 1) * sd / np.sqrt(nn)
            print(f"  {mname:<12} mean {v.mean():+.4f}  sd {sd:.4f}  favour MG {int((v>0).sum())}/{nn}  "
                  f"t p {pv:.4f}  95% CI [{v.mean()-half:+.4f},{v.mean()+half:+.4f}]")
            part3.append(dict(direction=dlab, metric=mname, subset_n_per_seed=";".join(str(per_seed[s][0]) for s in SEEDS),
                              per_seed=";".join(f"{s}:{per_seed[s][1][mk]:+.4f}" for s in SEEDS),
                              mean_delta=v.mean(), sd=sd, n_favour_mg=int((v > 0).sum()), t_p=pv,
                              ci_lo=v.mean()-half, ci_hi=v.mean()+half))
    pd.DataFrame(part3).to_csv(os.path.join(R, "conditioned_rank1_deltas_6seed.csv"), index=False)

    # ---------------- PART 4 ----------------
    print("\n" + "=" * 110); print("PART 4: ORACLE UPPER BOUND (rank by r_ij itself; label-only)"); print("=" * 110)
    p4 = []
    for lab, lpath in (("CheXbert", CHEXBERT), ("CheXpert", CHEXPERT)):
        Rm, _ = relevance(ids_ref, lpath)
        tot = (Rm > 0).sum(axis=1); rst = tot > 0
        ideal = (-np.sort(-Rm, axis=1))[:, :TOP]
        n = len(ids_ref)
        oracle = torch.from_numpy(Rm.astype(np.float32)).to(device)   # similarity == relevance
        n5, n10, ap, p5 = scoring.compute_metrics_for_direction(oracle, Rm, ideal, tot, n)
        del oracle; torch.cuda.empty_cache()
        print(f"\n  {lab}: restricted n={int(rst.sum())}  (R is symmetric, so I->T and T->I are identical)")
        for mname, arr in (("nDCG@5", n5), ("nDCG@10", n10), ("mAP@10", ap), ("Precision@5", p5)):
            v = float(arr[rst].mean())
            for dlab in ("Image->Text", "Text->Image"):
                p4.append(dict(labeler=lab, direction=dlab, metric=mname, oracle_value=v, restricted_n=int(rst.sum())))
            print(f"    {mname:<12} = {v:.4f}")
    pd.DataFrame(p4).to_csv(os.path.join(R, "oracle_upper_bound_restricted.csv"), index=False)

    # ---------------- PART 2 ----------------
    print("\n" + "=" * 110); print("PART 2: QUALITATIVE EXAMPLES (seed 3407, I->T, CheXbert restricted)"); print("=" * 110)
    a = store[("Paper 1", 3407)]["i2t"]; b = store[("MG-G2L", 3407)]["i2t"]
    keep = restricted & (a["paired_rank"] == 1) & (b["paired_rank"] == 1)
    delta = b["per"]["ndcg10"] - a["per"]["ndcg10"]
    kd = delta[keep]; kidx = np.where(keep)[0]
    q25, q50, q75 = np.percentile(kd, [25, 50, 75])
    print(f"queries kept = {int(keep.sum())}   delta percentiles: p25={q25:+.6f}  p50={q50:+.6f}  p75={q75:+.6f}")
    cb = pd.read_csv(CHEXBERT, dtype={"study_id": str}).set_index("study_id")
    fcols = [c for c in cb.columns if c not in ("subject_id",) and c != "No Finding"]
    def pos(sid): return [c for c in fcols if cb.loc[str(sid), c] == 1]
    picks = {}
    for lbl, target in (("A (75th pct, typical gain)", q75), ("B (25th pct, typical loss)", q25)):
        cand = kidx[np.isclose(kd, target, atol=1e-12)]
        if len(cand) == 0:
            cand = kidx[[int(np.argmin(np.abs(kd - target)))]]
        pick = cand[np.argmin([int(ids_ref[c]) for c in cand])]
        picks[lbl] = (pick, len(cand))
    out = []
    for lbl, (i, nties) in picks.items():
        sid = ids_ref[i]
        qpos = pos(sid)
        nshare = int(((Rm_cb[i] > 0) & restricted).sum())
        print(f"\n### Example {lbl}")
        print(f"  query study_id={sid}  delta={delta[i]:+.6f}  (ties at this percentile: {nties})")
        print(f"  query CheXbert positives ({len(qpos)}): {qpos}")
        print(f"  restricted candidates sharing >=1 finding: {nshare}")
        for arm, st in (("Paper 1", a), ("MG-G2L", b)):
            print(f"  -- {arm}: nDCG@10={st['per']['ndcg10'][i]:.4f}  Precision@5={st['per']['prec5'][i]:.4f}")
            print(f"     {'rank':<5}|{'sim':<9}|{'study_id':<12}|{'r_ij':<5}|{'paired':<7}|positives")
            for k in range(5):
                j = st["idxs"][i][k]
                print(f"     {k+1:<5}|{st['sims'][i][k]:<9.4f}|{str(ids_ref[j]):<12}|{int(Rm_cb[i, j]):<5}|"
                      f"{str(j == i):<7}|{pos(ids_ref[j])}")
                out.append(dict(example=lbl, arm=arm, rank=k+1, query_study_id=str(sid), retrieved_study_id=str(ids_ref[j])))
    json.dump({"kept": int(keep.sum()), "p25": float(q25), "p50": float(q50), "p75": float(q75),
               "example_study_ids": {k: str(ids_ref[v[0]]) for k, v in picks.items()}},
              open(os.path.join(RK, "example_selection_seed3407.json"), "w"), indent=2)
    print("\nPART 2 ids written to mimic/results/rankings/example_selection_seed3407.json (gitignored)")
    print("\nDONE")


if __name__ == "__main__":
    main()
