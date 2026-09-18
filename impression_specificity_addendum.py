#!/usr/bin/env python3
"""
STEP A2 -- Impression-specificity addendum (CPU). For the MIMIC-CXR, ReXGradient
and Open-I (openi_sa) TEST splits, measures how template-like the Impression
sections are:

  * n (studies with a non-empty Impression), mean Impression length in tokens
  * type-token ratio (TTR) -- corpus-level, plus a size-matched TTR (random
    subsample to the smallest corpus, 20 repeats) because raw TTR falls with n
  * % of Impressions that are an EXACT duplicate of another Impression in the
    same split, and % with a token-Jaccard >= 0.9 near-duplicate
  * top-10 most frequent whole Impressions with their share of the split

Impression text comes from the SAME sources as the divergence scripts:
  MIMIC        mimic/scripts/compute_divergence_scores_paper2.extract_text_from_report
               on organized_data/reports, for every study in the MIMIC test label file
  ReXGradient  same extractor on the ReXGradient organized reports (hybrid_split=='test')
  openi_sa     texts_test_openi_sa.csv, column impression_text
Tokens: lowercase, strip non-alphanumerics, whitespace split (no stopword removal).

Outputs:
  mimic/results/impression_specificity_three_datasets.csv   (aggregate, committed)
  mimic/results/impression_top10_{dataset}.csv               (raw text -> gitignored)
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "mimic", "scripts"))
import compute_divergence_scores_paper2 as div  # noqa: E402  extractor reused unchanged

R = os.path.join(PROJECT_DIR, "mimic", "results")
MIMIC_LABELS = os.path.join(PROJECT_DIR, "mimic", "data", "test_labels_chexbert_binary.csv")
MIMIC_META = div.ORIGINAL_METADATA_CSV
MIMIC_REPORTS = div.REPORTS_DIR
REX_META = "/home/abedin/Developments/chest_x_ray_data_processing/rexgradient/data/rexgradient_metadata_mimic_layout.csv"
REX_REPORTS = "/home/abedin/Developments/download_ReXGradient-160K_python_file/raw_data_ReXGradient-160K/organized_data/reports"
OPENI_TEXTS = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/openi_sa/texts_test_openi_sa.csv"
JACCARD_T = 0.9
RNG = np.random.default_rng(0)


def norm_tokens(text):
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split()


def load_mimic():
    ids = pd.read_csv(MIMIC_LABELS, dtype={"study_id": str})["study_id"]
    meta = pd.read_csv(MIMIC_META, dtype=str).set_index("study_id")
    out, missing = [], 0
    for sid in ids:
        if sid not in meta.index:
            missing += 1; continue
        p = os.path.join(MIMIC_REPORTS, meta.loc[sid, "report_file"])
        if not os.path.exists(p):
            missing += 1; continue
        out.append(div.extract_text_from_report(p)[1])
    print(f"MIMIC: {len(ids)} test studies, {missing} without report file")
    return out


def load_rex():
    meta = pd.read_csv(REX_META, dtype=str)
    test = meta[meta["hybrid_split"] == "test"]
    out, missing = [], 0
    for rf in test["report_file"]:
        p = os.path.join(REX_REPORTS, rf)
        if not os.path.exists(p):
            missing += 1; continue
        out.append(div.extract_text_from_report(p)[1])
    print(f"ReXGradient: {len(test)} test studies, {missing} without report file")
    return out


def load_openi():
    t = pd.read_csv(OPENI_TEXTS, dtype={"study_id": str})
    print(f"openi_sa: {len(t)} test rows")
    return ["" if pd.isna(x) else str(x) for x in t["impression_text"]]


def near_dup_fraction(token_lists):
    """Fraction of docs having >=1 OTHER doc with token-set Jaccard >= JACCARD_T (chunked sparse)."""
    vocab = {}
    rows, cols = [], []
    for i, toks in enumerate(token_lists):
        for t in set(toks):
            rows.append(i); cols.append(vocab.setdefault(t, len(vocab)))
    n = len(token_lists)
    X = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, len(vocab)))
    sizes = np.asarray(X.sum(axis=1)).ravel()
    has = np.zeros(n, dtype=bool)
    step = 1000
    for s in range(0, n, step):
        inter = (X[s:s + step] @ X.T).toarray().astype(np.float64)
        union = sizes[s:s + step, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        jac[np.arange(inter.shape[0]), np.arange(s, s + inter.shape[0])] = 0.0  # exclude self
        has[s:s + step] = (jac >= JACCARD_T).any(axis=1)
    return float(has.mean())


def analyse(name, impressions, match_n):
    toks = [norm_tokens(x) for x in impressions]
    keep = [i for i, t in enumerate(toks) if len(t) > 0]
    toks = [toks[i] for i in keep]
    n = len(toks)
    lengths = np.array([len(t) for t in toks])
    all_tokens = [t for ts in toks for t in ts]
    ttr = len(set(all_tokens)) / len(all_tokens)
    # size-matched TTR
    mt = []
    for _ in range(20):
        idx = RNG.choice(n, size=min(match_n, n), replace=False)
        sub = [t for i in idx for t in toks[i]]
        mt.append(len(set(sub)) / len(sub))
    norm_text = [" ".join(t) for t in toks]
    vc = pd.Series(norm_text).value_counts()
    exact_dup_frac = float(sum(c for c in vc.values if c >= 2) / n)
    jac_frac = near_dup_fraction(toks)
    top = vc.head(10)
    top_df = pd.DataFrame({"rank": range(1, len(top) + 1), "impression_normalised": top.index,
                           "count": top.values, "share_pct": 100 * top.values / n})
    top_df.to_csv(os.path.join(R, f"impression_top10_{name}.csv"), index=False)
    row = {"dataset": name, "n_with_impression": n, "n_input_rows": len(impressions),
           "mean_impression_tokens": float(lengths.mean()), "median_impression_tokens": float(np.median(lengths)),
           "ttr_corpus": ttr, "ttr_size_matched": float(np.mean(mt)), "size_matched_n": min(match_n, n),
           "n_unique_impressions": int(len(vc)), "pct_exact_duplicate": 100 * exact_dup_frac,
           "pct_jaccard_ge_0.9_neardup": 100 * jac_frac,
           "top1_share_pct": float(100 * top.values[0] / n), "top10_cum_share_pct": float(100 * top.values.sum() / n)}
    print(f"\n=== {name}: n={n} (of {len(impressions)} rows) ===")
    for k, v in row.items():
        if k != "dataset":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  top-10 whole Impressions (normalised):")
    for _, r in top_df.iterrows():
        print(f"    {r['rank']:>2}. {r['share_pct']:5.2f}%  ({r['count']:>5})  {r['impression_normalised'][:110]}")
    return row


def main():
    out = os.path.join(R, "impression_specificity_three_datasets.csv")
    assert not os.path.exists(out), f"refusing to overwrite {out}"
    data = {"mimic": load_mimic(), "rexgradient": load_rex(), "openi_sa": load_openi()}
    match_n = min(sum(1 for x in v if norm_tokens(x)) for v in data.values())
    rows = [analyse(k, v, match_n) for k, v in data.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("\n===== FINAL SUMMARY (A2 IMPRESSION SPECIFICITY) =====")
    for _, r in df.iterrows():
        print(f"{r.dataset}: n={r.n_with_impression} mean_len={r.mean_impression_tokens:.1f} ttr={r.ttr_corpus:.3f} "
              f"ttr_matched(n={r.size_matched_n})={r.ttr_size_matched:.3f} exact_dup={r['pct_exact_duplicate']:.1f}% "
              f"jaccard>=0.9={r['pct_jaccard_ge_0.9_neardup']:.1f}% top1={r.top1_share_pct:.1f}% top10={r.top10_cum_share_pct:.1f}%")
    print(f"csv={out}")
    print("===== END SUMMARY =====")


if __name__ == "__main__":
    main()
