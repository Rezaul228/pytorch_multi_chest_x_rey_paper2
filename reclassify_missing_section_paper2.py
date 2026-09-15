#!/usr/bin/env python3
"""
Re-emit the missing-section verdict (single_section vs both_sections, I2T,
restricted subset) using Mann-Whitney U direction + p-value and group MEAN
deltas, instead of the strict median inequality (which is invalid for the
0.2-quantized Precision@5). Pure statistics on cached per-query CSVs.
"""

import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from extend_hypothesis_tests_6seed_paper2 import (  # noqa: E402
    ALL_SEEDS, METRICS, load_and_verify_per_query, get_restricted_ids_and_section_group,
)

ALPHA = 0.05


def main():
    restricted_ids, section_group = get_restricted_ids_and_section_group()

    print("seed | metric | mean_single | mean_both | U_pvalue | direction (single>both? yes/no) | verdict")
    per_seed_rows = {}
    for seed in ALL_SEEDS:
        df = load_and_verify_per_query(seed)
        sub = df[df["study_id"].isin(restricted_ids)].copy()
        sub["group"] = sub["study_id"].apply(section_group)
        assert (sub["group"] != "neither").all()

        per_seed_rows[seed] = []
        for label, key in METRICS:
            delta = (sub[f"mg_{key}"] - sub[f"p1_{key}"]).values
            single = delta[(sub["group"] == "single_section").values]
            both = delta[(sub["group"] == "both_sections").values]

            _, p_two = mannwhitneyu(single, both, alternative="two-sided")
            _, p_greater = mannwhitneyu(single, both, alternative="greater")
            _, p_less = mannwhitneyu(single, both, alternative="less")
            direction_yes = p_greater < p_less

            if p_two < ALPHA and direction_yes:
                verdict = "support"
            elif p_two < ALPHA and not direction_yes:
                verdict = "contradict"
            else:
                verdict = "neutral"
            per_seed_rows[seed].append(verdict)

            print(f"{seed} | {label} | {single.mean():+.4f} | {both.mean():+.4f} | {p_two:.2e} | "
                  f"{'yes' if direction_yes else 'no'} | {verdict}")

    support = neutral = contradict = 0
    seed_verdicts = {}
    for seed, rows in per_seed_rows.items():
        if "contradict" in rows:
            v = "contradict"; contradict += 1
        elif "support" in rows:
            v = "support"; support += 1
        else:
            v = "neutral"; neutral += 1
        seed_verdicts[seed] = v

    detail = ", ".join(f"{s}={v}" for s, v in seed_verdicts.items())
    print(f"\nOverall tally (6 seeds): {support} support / {neutral} neutral / {contradict} contradict  [{detail}]")


if __name__ == "__main__":
    main()
