#!/usr/bin/env python3
"""
Laterality/severity tagging for hard-negative mining candidates, on the
12,429-sample verified MIMIC-CXR TEST set. Text-only, CPU-only, run
interactively (lightweight, independent of the running train/val SLURM job
113645 -- reads its own report files, does not touch that job's output).

For each of 6 findings that commonly carry laterality/severity in
radiology reports (Pleural Effusion, Pneumothorax, Atelectasis, Lung
Opacity, Edema, Consolidation), for every study_id where
test_labels_chexpert_binary.csv marks that finding positive:
  - re-extract findings+impression text from the raw report file (the
    section-boundary work only ever saved token COUNTS, not the raw text
    itself, so there is no saved intermediate to reuse -- this re-reads
    report files directly, same as build_section_boundaries_paper2.py did)
  - find the FIRST mention of the finding's name (or a listed synonym) in
    the combined text
  - within a +/-60-character window around that mention, tag laterality
    (RIGHT / LEFT / BILATERAL / UNSPECIFIED) and severity (one of
    mild/moderate/severe/small/large/trace, or UNSPECIFIED)

Laterality rule: BILATERAL if "bilateral" appears in the window, OR if
both "right" and "left" appear in the window; else RIGHT or LEFT if only
one appears; else UNSPECIFIED.
Severity rule: if multiple severity keywords appear in the window, priority
order is severe > moderate > mild > large > small > trace (first match in
that priority order is used); else UNSPECIFIED.

If the finding keyword itself is never found in the text (can happen --
CheXbert/NegBio labels are derived from the full report by a separate NLP
labeler and may not always match a literal substring), the pair is still
recorded with UNSPECIFIED/UNSPECIFIED and counted separately as
"keyword not found" for transparency.

Output: hard_negative_tags_test_paper2.csv
  columns: study_id, finding, laterality_tag, severity_tag
"""

import os
import re
import csv

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

BINARY_LABELS_PATH = os.path.join(PROJECT_DIR, "test_labels_chexpert_binary.csv")
ORIGINAL_METADATA_CSV = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
REPORTS_DIR = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"
OUTPUT_CSV_PATH = os.path.join(PROJECT_DIR, "hard_negative_tags_test_paper2.csv")

FINDING_SEARCH_TERMS = {
    'Pleural Effusion': ['pleural effusion', 'effusion'],
    'Pneumothorax': ['pneumothorax'],
    'Atelectasis': ['atelectasis'],
    'Lung Opacity': ['opacity', 'opacities'],
    'Edema': ['edema'],
    'Consolidation': ['consolidation'],
}

LATERALITY_KEYWORDS = ['right', 'left', 'bilateral']
SEVERITY_KEYWORDS_PRIORITY = ['severe', 'moderate', 'mild', 'large', 'small', 'trace']

WINDOW_CHARS = 60


def extract_text_from_report(report_path):
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


def combine_findings_impression(findings, impression):
    combined_text = ""
    if findings:
        combined_text += findings
    if impression:
        if combined_text:
            combined_text += " " + impression
        else:
            combined_text = impression
    return combined_text


def find_first_mention_window(text_lower, search_terms, window_chars):
    """Returns the +/-window_chars substring around the FIRST occurrence of
    any of search_terms in text_lower, or None if none found."""
    best_pos = None
    for term in search_terms:
        pos = text_lower.find(term)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos = pos
    if best_pos is None:
        return None
    start = max(0, best_pos - window_chars)
    end = min(len(text_lower), best_pos + window_chars)
    return text_lower[start:end]


def tag_laterality(window_text):
    has_right = re.search(r'\bright\b', window_text) is not None
    has_left = re.search(r'\bleft\b', window_text) is not None
    has_bilateral = re.search(r'\bbilateral\b', window_text) is not None

    if has_bilateral or (has_right and has_left):
        return 'BILATERAL'
    if has_right:
        return 'RIGHT'
    if has_left:
        return 'LEFT'
    return 'UNSPECIFIED'


def tag_severity(window_text):
    for keyword in SEVERITY_KEYWORDS_PRIORITY:
        if re.search(rf'\b{keyword}\b', window_text):
            return keyword.upper()
    return 'UNSPECIFIED'


def main():
    print("Loading test_labels_chexpert_binary.csv...")
    binary_df = pd.read_csv(BINARY_LABELS_PATH, dtype={'study_id': str}).set_index('study_id')

    print("Loading original metadata CSV for report_file lookup...")
    meta_df = pd.read_csv(ORIGINAL_METADATA_CSV)
    meta_df = meta_df.set_index(meta_df['study_id'].astype(str))

    text_cache = {}  # study_id -> combined lowercased text (avoid re-reading a report for multiple findings)

    def get_combined_text_lower(sid):
        if sid in text_cache:
            return text_cache[sid]
        if sid not in meta_df.index:
            text_cache[sid] = None
            return None
        report_file = meta_df.loc[sid, 'report_file']
        report_path = os.path.join(REPORTS_DIR, report_file)
        if not os.path.exists(report_path):
            text_cache[sid] = None
            return None
        findings, impression = extract_text_from_report(report_path)
        combined = combine_findings_impression(findings, impression)
        combined_lower = combined.lower()
        text_cache[sid] = combined_lower
        return combined_lower

    rows_out = []
    keyword_not_found_count = 0
    missing_report_count = 0

    for finding, search_terms in FINDING_SEARCH_TERMS.items():
        positive_study_ids = binary_df.index[binary_df[finding] == 1].tolist()
        print(f"\n{finding}: {len(positive_study_ids)} positive study_ids")

        for sid in positive_study_ids:
            text_lower = get_combined_text_lower(sid)
            if text_lower is None:
                missing_report_count += 1
                rows_out.append({'study_id': sid, 'finding': finding,
                                  'laterality_tag': 'UNSPECIFIED', 'severity_tag': 'UNSPECIFIED'})
                continue

            window = find_first_mention_window(text_lower, search_terms, WINDOW_CHARS)
            if window is None:
                keyword_not_found_count += 1
                rows_out.append({'study_id': sid, 'finding': finding,
                                  'laterality_tag': 'UNSPECIFIED', 'severity_tag': 'UNSPECIFIED'})
                continue

            laterality_tag = tag_laterality(window)
            severity_tag = tag_severity(window)
            rows_out.append({'study_id': sid, 'finding': finding,
                              'laterality_tag': laterality_tag, 'severity_tag': severity_tag})

    print(f"\nTotal (study_id, finding) pairs tagged: {len(rows_out)}")
    print(f"  Report file missing entirely: {missing_report_count}")
    print(f"  Finding keyword not found in text (fell back to UNSPECIFIED/UNSPECIFIED): {keyword_not_found_count}")

    with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['study_id', 'finding', 'laterality_tag', 'severity_tag'])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
    print(f"\nSaved: {OUTPUT_CSV_PATH}  ({len(rows_out)} rows)")

    out_df = pd.DataFrame(rows_out)

    print("\n" + "=" * 100)
    print("SUMMARY: laterality tag counts per finding")
    print("=" * 100)
    lat_summary = out_df.groupby(['finding', 'laterality_tag']).size().unstack(fill_value=0)
    lat_cols = [c for c in ['RIGHT', 'LEFT', 'BILATERAL', 'UNSPECIFIED'] if c in lat_summary.columns]
    print(lat_summary[lat_cols].to_string())

    print("\n" + "=" * 100)
    print("SUMMARY: severity tag counts per finding")
    print("=" * 100)
    sev_summary = out_df.groupby(['finding', 'severity_tag']).size().unstack(fill_value=0)
    sev_cols = [c for c in ['SEVERE', 'MODERATE', 'MILD', 'LARGE', 'SMALL', 'TRACE', 'UNSPECIFIED'] if c in sev_summary.columns]
    print(sev_summary[sev_cols].to_string())

    print("\n" + "=" * 100)
    print("HARD-NEGATIVE POOL SIZE: RIGHT vs LEFT pairs (same finding, different study_ids)")
    print("=" * 100)
    for finding in FINDING_SEARCH_TERMS:
        sub = out_df[out_df['finding'] == finding]
        n_right = (sub['laterality_tag'] == 'RIGHT').sum()
        n_left = (sub['laterality_tag'] == 'LEFT').sum()
        n_pairs = int(n_right) * int(n_left)
        print(f"  {finding}: {n_right} RIGHT vs {n_left} LEFT -> {n_pairs:,} potential hard-negative (RIGHT,LEFT) pairs")


if __name__ == "__main__":
    main()
