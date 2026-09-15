#!/usr/bin/env python3
"""
Findings/Impression "divergence" scoring for the 12,429-sample MIMIC-CXR
test set. Text-only, CPU-only.

For each report restricted to has_findings=True AND has_impression=True
AND >=1 confirmed-positive CheXpert finding (excluding "No Finding"):
  - for each confirmed-positive finding, check (case-insensitive substring)
    whether ANY of its keyword variants appear in the Findings text, the
    Impression text, or both
  - divergence_score = count of findings whose keyword appears in exactly
    ONE of the two sections (not both, and not neither)

Reports are then split into:
  - "agreement":  divergence_score == 0
  - "divergence": divergence_score >= 1

Findings/Impression text is NOT cached anywhere from earlier work (only
token counts/booleans were ever saved) -- re-extracted from the raw report
files, same source/logic as build_section_boundaries_paper2.py and
build_hard_negative_tags_paper2.py.

Output: divergence_scores_test_paper2.csv
  columns: study_id, divergence_score, group
  (only for the restricted population defined above)
"""

import os
import re
import csv
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import paths
PROJECT_DIR = paths.repo_root()
MIMIC_DATA_DIR = os.path.join(PROJECT_DIR, "mimic", "data")
MIMIC_RESULTS_DIR = os.path.join(PROJECT_DIR, "mimic", "results")

BINARY_LABELS_PATH = os.path.join(MIMIC_DATA_DIR, "test_labels_chexpert_binary.csv")
SECTION_BOUNDARIES_PATH = os.path.join(MIMIC_DATA_DIR, "section_boundaries_test_paper2.csv")
ORIGINAL_METADATA_CSV = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
REPORTS_DIR = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"
OUTPUT_CSV_PATH = os.path.join(MIMIC_RESULTS_DIR, "divergence_scores_test_paper2.csv")

# Documented keyword mapping for the 13 non-"No Finding" CheXpert categories.
# Chosen to be sensible/common radiology phrasing for each category; not an
# exhaustive clinical NLP lexicon. Case-insensitive substring match.
FINDING_KEYWORDS = {
    'Atelectasis': ['atelectasis'],
    'Cardiomegaly': ['cardiomegaly', 'heart size', 'enlarged heart', 'cardiac silhouette'],
    'Consolidation': ['consolidation'],
    'Edema': ['edema'],
    'Enlarged Cardiomediastinum': ['mediastinum', 'mediastinal', 'widened mediastinum'],
    'Fracture': ['fracture'],
    'Lung Lesion': ['lesion', 'nodule', 'mass'],
    'Lung Opacity': ['opacity', 'opacities'],
    'Pleural Effusion': ['pleural effusion', 'effusion'],
    'Pleural Other': ['pleural thickening', 'pleural scarring', 'fibrothorax'],
    'Pneumonia': ['pneumonia'],
    'Pneumothorax': ['pneumothorax'],
    'Support Devices': ['tube', 'catheter', 'line', 'pacemaker', 'wire', 'lead', 'device'],
}


def extract_text_from_report(report_path):
    """Same regex logic used throughout this project's section-boundary/
    hard-negative-tagging scripts."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: could not read report {report_path}: {e}")
        return "", ""

    findings = ""
    impression = ""
    findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', content, re.DOTALL | re.IGNORECASE)
    if findings_match:
        findings = findings_match.group(1).strip()
    impression_match = re.search(r'IMPRESSION:(.*?)$', content, re.DOTALL | re.IGNORECASE)
    if impression_match:
        impression = impression_match.group(1).strip()
    return findings, impression


def keyword_present(text_lower, keywords):
    return any(kw in text_lower for kw in keywords)


def main():
    print("Loading test_labels_chexpert_binary.csv...")
    binary_df = pd.read_csv(BINARY_LABELS_PATH, dtype={'study_id': str}).set_index('study_id')
    finding_cols = [c for c in binary_df.columns if c not in ('subject_id',) and c != 'No Finding']
    assert len(finding_cols) == 13, f"expected 13 findings, got {len(finding_cols)}"
    assert set(finding_cols) == set(FINDING_KEYWORDS.keys()), (
        f"keyword mapping mismatch: {set(finding_cols) ^ set(FINDING_KEYWORDS.keys())}"
    )

    print("Loading section_boundaries_test_paper2.csv (has_findings/has_impression flags)...")
    boundaries_df = pd.read_csv(SECTION_BOUNDARIES_PATH, dtype={'study_id': str}).set_index('study_id')

    print("Loading original metadata CSV for report_file lookup...")
    meta_df = pd.read_csv(ORIGINAL_METADATA_CSV)
    meta_df = meta_df.set_index(meta_df['study_id'].astype(str))

    rows_out = []
    n_total = len(binary_df)
    n_restricted = 0
    n_skipped_no_report = 0

    for sid in binary_df.index:
        if sid not in boundaries_df.index:
            continue
        has_find = bool(boundaries_df.loc[sid, 'has_findings'])
        has_imp = bool(boundaries_df.loc[sid, 'has_impression'])
        if not (has_find and has_imp):
            continue

        positive_findings = [c for c in finding_cols if binary_df.loc[sid, c] == 1]
        if len(positive_findings) == 0:
            continue

        n_restricted += 1

        if sid not in meta_df.index:
            n_skipped_no_report += 1
            continue
        report_file = meta_df.loc[sid, 'report_file']
        report_path = os.path.join(REPORTS_DIR, report_file)
        if not os.path.exists(report_path):
            n_skipped_no_report += 1
            continue

        findings_text, impression_text = extract_text_from_report(report_path)
        findings_lower = findings_text.lower()
        impression_lower = impression_text.lower()

        divergence_score = 0
        for finding in positive_findings:
            keywords = FINDING_KEYWORDS[finding]
            in_findings = keyword_present(findings_lower, keywords)
            in_impression = keyword_present(impression_lower, keywords)
            if in_findings != in_impression:  # exactly one of the two -- XOR
                divergence_score += 1

        group = 'agreement' if divergence_score == 0 else 'divergence'
        rows_out.append({'study_id': sid, 'divergence_score': divergence_score, 'group': group})

    print(f"\nTotal test study_ids: {n_total}")
    print(f"Restricted population (has_find AND has_imp AND >=1 positive finding): {n_restricted}")
    if n_skipped_no_report:
        print(f"Warning: {n_skipped_no_report} restricted study_ids skipped (report file missing)")

    n_agreement = sum(1 for r in rows_out if r['group'] == 'agreement')
    n_divergence = sum(1 for r in rows_out if r['group'] == 'divergence')
    print(f"\nGroup sizes:")
    print(f"  agreement  (divergence_score == 0): {n_agreement}")
    print(f"  divergence (divergence_score >= 1): {n_divergence}")
    print(f"  total scored: {len(rows_out)}")

    from collections import Counter
    score_dist = Counter(r['divergence_score'] for r in rows_out)
    print(f"\nDivergence score distribution: {dict(sorted(score_dist.items()))}")

    with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['study_id', 'divergence_score', 'group'])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
    print(f"\nSaved: {OUTPUT_CSV_PATH}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
