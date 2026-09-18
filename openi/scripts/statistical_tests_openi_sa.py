#!/usr/bin/env python3
"""
Open-I (openi_sa) statistics on the per-query caches from
evaluate_test_openi_sa.py. CPU only; reads nothing from mimic/.

PRIMARY test is SEED-LEVEL (n = 6 seeds, not n = queries):
  paired t-test + exact two-sided sign test on the 6 per-seed mean deltas,
  per metric and direction, plus a 10,000-resample bootstrap 95% CI.
SECONDARY is the per-query Wilcoxon, explicitly labelled underpowered, with the
post-hoc minimum detectable effect (MDE) at 80% power so a null is not read as
"no effect".

Every aggregate is reported TWICE:
  (a) all 6 seeds
  (b) the 5 seed-pairs where BOTH arms converged -- Paper 1 seed 1337 reached
      best val R@1 0.8352 while every other run reached 0.98-1.00, so a delta
      against it measures that training failure, not the architecture.
Seed 1337 is shown, never silently dropped.

Missing-section analysis (the reason Open-I is in the paper):
  96 impression-only vs 656 both-section test queries. Reports how many of each
  survive the restricted-subset filter and the two-sample MDE BEFORE any test,
  then Mann-Whitney U on the per-query deltas between groups, group mean deltas,
  and the MIMIC baseline-headroom control: delta / mean(1 - Paper 1 score).
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, norm, t as tdist, ttest_1samp, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
P = paths.repo_root()
R = os.path.join(P, "openi", "results")
LABELS = os.path.join(P, "openi", "data", "test_labels_chexbert_binary.csv")
BOUNDARIES = os.path.join(P, "openi", "data", "section_boundaries_test_openi_sa.csv")

SEEDS = [17, 42, 123, 1337, 2021, 3407]
UNCONVERGED = [1337]                       # Paper 1 seed 1337: best val R@1 0.8352
CONVERGED = [s for s in SEEDS if s not in UNCONVERGED]
METRICS = [("nDCG@5", "ndcg5"), ("nDCG@10", "ndcg10"), ("mAP@10", "map10"), ("Precision@5", "prec5")]
DIRS = [("Image->Text", "i2t"), ("Text->Image", "t2i")]
RNG = np.random.default_rng(0)
TRUE = ("1", "True", "true")


def restricted_ids():
    rows = list(csv.DictReader(open(LABELS)))
    cols = [c for c in rows[0] if c not in ("subject_id", "study_id") and c != "No Finding"]
    assert len(cols) == 13, cols
    return {r["study_id"] for r in rows if any(r[c] == "1" for c in cols)}


def section_groups():
    rows = list(csv.DictReader(open(BOUNDARIES)))
    both = {r["study_id"] for r in rows if r["has_findings"] in TRUE and r["has_impression"] in TRUE}
    imp = {r["study_id"] for r in rows if r["has_findings"] not in TRUE and r["has_impression"] in TRUE}
    return both, imp


def mde_paired(sd_diff, n, power=0.80, alpha=0.05):
    """Smallest mean paired difference detectable at `power`, two-sided."""
    if n < 2 or not np.isfinite(sd_diff) or sd_diff == 0:
        return float("nan")
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd_diff / np.sqrt(n)


def mde_two_sample(sd_pooled, n1, n2, power=0.80, alpha=0.05):
    if min(n1, n2) < 2 or not np.isfinite(sd_pooled) or sd_pooled == 0:
        return float("nan")
    return (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd_pooled * np.sqrt(1 / n1 + 1 / n2)


def boot_ci(d, n_boot=10000):
    b = np.array([RNG.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    for f in ["seed_level_tests_openi_sa.csv", "per_query_wilcoxon_openi_sa.csv",
              "missing_section_openi_sa.csv", "verdict_openi_sa.json"]:
        assert not os.path.exists(os.path.join(R, f)), f"refusing to overwrite {f}"

    rid = restricted_ids()
    both_ids, imp_ids = section_groups()
    per = {s: pd.read_csv(os.path.join(R, f"per_query_metrics_openi_sa_seed_{s}.csv"), dtype={"study_id": str})
           for s in SEEDS}
    n_rows = len(per[SEEDS[0]])
    sub = {s: d[d.study_id.isin(rid)] for s, d in per.items()}
    rn = len(sub[SEEDS[0]])
    assert all(len(v) == rn for v in sub.values())
    print(f"test rows={n_rows}  restricted n={rn}  (both-section={len(both_ids)}, impression-only={len(imp_ids)})")

    # ---------------- seed-level (PRIMARY) + per-query Wilcoxon (SECONDARY)
    seed_rows, wil_rows = [], []
    n_tests = len(METRICS) * len(DIRS) * len(SEEDS)
    for dname, dk in DIRS:
        for mname, mk in METRICS:
            deltas = {}
            for s in SEEDS:
                d = sub[s]
                p1, mg = d[f"p1_{mk}_{dk}"].values, d[f"mg_{mk}_{dk}"].values
                diff = mg - p1
                deltas[s] = float(diff.mean())
                nz = int((diff != 0).sum())
                stat, pv = (wilcoxon(mg, p1, zero_method="wilcox", alternative="two-sided")
                            if nz else (np.nan, np.nan))
                wil_rows.append({"direction": dname, "metric": mname, "seed": s, "n": len(d),
                                 "mean_delta": float(diff.mean()), "p_raw": float(pv),
                                 "p_bonferroni": float(min(1.0, pv * n_tests)) if np.isfinite(pv) else np.nan,
                                 "sig_raw": bool(np.isfinite(pv) and pv < 0.05),
                                 "n_improved": int((diff > 0).sum()), "n_worse": int((diff < 0).sum()),
                                 "mde_80pct_power": float(mde_paired(diff.std(ddof=1), len(diff)))})
            for label, ss in [("all6", SEEDS), ("conv5", CONVERGED)]:
                d = np.array([deltas[s] for s in ss])
                npos = int((d > 0).sum())
                sign_p = binomtest(npos, len(d), 0.5, alternative="two-sided").pvalue if np.all(d != 0) else np.nan
                lo, hi = boot_ci(d)
                seed_rows.append({"subset": label, "direction": dname, "metric": mname, "n_seeds": len(d),
                                  "mean_delta": float(d.mean()), "sd_delta": float(d.std(ddof=1)),
                                  "n_positive": npos, "sign_test_p": float(sign_p),
                                  "paired_t_p": float(ttest_1samp(d, 0.0).pvalue),
                                  "boot_ci_lo": lo, "boot_ci_hi": hi,
                                  "mde_80pct_power": float(mde_paired(d.std(ddof=1), len(d))),
                                  "per_seed_deltas": ";".join(f"{s}:{deltas[s]:+.4f}" for s in ss)})
    sl = pd.DataFrame(seed_rows); sl.to_csv(os.path.join(R, "seed_level_tests_openi_sa.csv"), index=False)
    wl = pd.DataFrame(wil_rows); wl.to_csv(os.path.join(R, "per_query_wilcoxon_openi_sa.csv"), index=False)

    # ---------------- missing-section analysis
    ms_rows = []
    for mname, mk in METRICS:
        for dname, dk in DIRS:
            for label, ss in [("all6", SEEDS), ("conv5", CONVERGED)]:
                gm, gp, hn = {}, {}, {}
                us, ps = [], []
                for s in ss:
                    d = sub[s]
                    d = d.assign(delta=d[f"mg_{mk}_{dk}"] - d[f"p1_{mk}_{dk}"])
                    a = d[d.study_id.isin(imp_ids)]; b = d[d.study_id.isin(both_ids)]
                    if len(a) < 3 or len(b) < 3:
                        continue
                    u, p = mannwhitneyu(a.delta.values, b.delta.values, alternative="two-sided")
                    us.append(float(u)); ps.append(float(p))
                    gm.setdefault("imp", []).append(float(a.delta.mean()))
                    gm.setdefault("both", []).append(float(b.delta.mean()))
                    gp.setdefault("imp", []).append(float(a[f"p1_{mk}_{dk}"].mean()))
                    gp.setdefault("both", []).append(float(b[f"p1_{mk}_{dk}"].mean()))
                    for g, df_ in (("imp", a), ("both", b)):
                        head = float((1 - df_[f"p1_{mk}_{dk}"]).mean())
                        hn.setdefault(g, []).append(float(df_.delta.mean()) / head if head > 0 else np.nan)
                if not us:
                    continue
                d0 = sub[ss[0]].assign(delta=sub[ss[0]][f"mg_{mk}_{dk}"] - sub[ss[0]][f"p1_{mk}_{dk}"])
                na = int(d0.study_id.isin(imp_ids).sum()); nb = int(d0.study_id.isin(both_ids).sum())
                sd_p = float(np.sqrt((d0[d0.study_id.isin(imp_ids)].delta.var(ddof=1) * (na - 1) +
                                      d0[d0.study_id.isin(both_ids)].delta.var(ddof=1) * (nb - 1)) / (na + nb - 2)))
                ms_rows.append({"subset": label, "metric": mname, "direction": dname,
                                "n_impression_only_restricted": na, "n_both_restricted": nb,
                                "mde_two_sample_80pct": float(mde_two_sample(sd_p, na, nb)),
                                "mean_delta_impression_only": float(np.mean(gm["imp"])),
                                "mean_delta_both": float(np.mean(gm["both"])),
                                "p1_mean_impression_only": float(np.mean(gp["imp"])),
                                "p1_mean_both": float(np.mean(gp["both"])),
                                "headroom_norm_impression_only": float(np.nanmean(hn["imp"])),
                                "headroom_norm_both": float(np.nanmean(hn["both"])),
                                "mannwhitney_p_median": float(np.median(ps)),
                                "n_seeds_p_lt_05": int(sum(p < 0.05 for p in ps)), "n_seeds": len(ps)})
    ms = pd.DataFrame(ms_rows); ms.to_csv(os.path.join(R, "missing_section_openi_sa.csv"), index=False)

    # ---------------- print
    print("\n===== FINAL SUMMARY (OPEN-I STATISTICS) =====")
    for label in ("all6", "conv5"):
        print(f"\n-- SEED-LEVEL (PRIMARY), subset={label} "
              f"({'all 6 seeds' if label == 'all6' else 'excl. Paper 1 seed 1337, unconverged'}) --")
        print(f"{'dir':<12} {'metric':<12} {'mean_delta':>11} {'x/n pos':>8} {'sign_p':>8} {'t_p':>8} "
              f"{'boot95%CI':>22} {'MDE80':>9}")
        for _, r in sl[sl.subset == label].iterrows():
            print(f"{r.direction:<12} {r.metric:<12} {r.mean_delta:+11.4f} {r.n_positive:>4}/{r.n_seeds:<3} "
                  f"{r.sign_test_p:8.3f} {r.paired_t_p:8.3f} [{r.boot_ci_lo:+.4f},{r.boot_ci_hi:+.4f}] {r.mde_80pct_power:9.4f}")
    print(f"\n-- PER-QUERY WILCOXON (SECONDARY, UNDERPOWERED: restricted n={rn}) --")
    for dname, _ in DIRS:
        s = wl[wl.direction == dname]
        print(f"{dname}: raw p<0.05 in {int(s.sig_raw.sum())}/{len(s)} (seed,metric) tests; "
              f"Bonferroni-significant {int((s.p_bonferroni < 0.05).sum())}/{len(s)}; "
              f"median per-test MDE at 80% power = {s.mde_80pct_power.median():.4f}")
    print(f"\n-- MISSING-SECTION (impression-only vs both-section, restricted) --")
    print(f"{'subset':<7} {'metric':<12} {'dir':<12} {'n_imp':>6} {'n_both':>7} {'MDE':>8} {'d_imp':>8} {'d_both':>8} "
          f"{'hn_imp':>8} {'hn_both':>8} {'U p(med)':>9} {'p<.05':>6}")
    for _, r in ms.iterrows():
        print(f"{r.subset:<7} {r.metric:<12} {r.direction:<12} {r.n_impression_only_restricted:>6} "
              f"{r.n_both_restricted:>7} {r.mde_two_sample_80pct:8.4f} {r.mean_delta_impression_only:+8.4f} "
              f"{r.mean_delta_both:+8.4f} {r.headroom_norm_impression_only:+8.4f} {r.headroom_norm_both:+8.4f} "
              f"{r.mannwhitney_p_median:9.3f} {r.n_seeds_p_lt_05:>3}/{r.n_seeds}")

    i2t5 = sl[(sl.subset == "conv5") & (sl.direction == "Image->Text")]
    verdict = {"restricted_n": rn, "n_impression_only_restricted": int(ms.n_impression_only_restricted.iloc[0]) if len(ms) else None,
               "n_both_restricted": int(ms.n_both_restricted.iloc[0]) if len(ms) else None,
               "primary_test": "seed-level paired t + sign test on 6 (or 5) per-seed deltas",
               "i2t_conv5": {r.metric: {"mean_delta": r.mean_delta, "sign_p": r.sign_test_p, "t_p": r.paired_t_p,
                                        "ci": [r.boot_ci_lo, r.boot_ci_hi], "mde": r.mde_80pct_power}
                             for _, r in i2t5.iterrows()},
               "any_i2t_significant_conv5": bool((i2t5.sign_test_p < 0.05).any() or (i2t5.paired_t_p < 0.05).any())}
    json.dump(verdict, open(os.path.join(R, "verdict_openi_sa.json"), "w"), indent=2)
    print(f"\nany I->T seed-level test significant (conv5)? {verdict['any_i2t_significant_conv5']}")
    print("===== END SUMMARY =====")


if __name__ == "__main__":
    main()
