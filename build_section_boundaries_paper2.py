#!/usr/bin/env python3
"""
Build a per-study lookup of findings-only token counts for the verified
12,429-sample MIMIC-CXR test set (mimic_shards_hybrid_full_ori).

Read-only w.r.t. the model and w.r.t. data_loader_v1.py: does not modify
either. Does NOT use IndianaDataLoader (which would load full images into
memory) -- instead reads the 'study_ids' field directly out of each test
shard pickle (cheap: no images, no captions materialized).

Tokenization reproduces the exact preprocessing used when this project's
vocab/shards were built (verified empirically against a known example --
study_id 53957785's first 20 caption token IDs decode to a token sequence
that exactly matches: lowercase -> regex-clean (keep word chars/whitespace/
hyphen/period) -> a few medical-abbreviation substitutions -> whitespace
split (NOT nltk.word_tokenize -- periods stay attached to words, e.g.
"normal.", which word_tokenize would split off) -> stopword removal via
nltk's English stopword list, EXCEPT a fixed set of medical/negation words
that are kept -> drop tokens of length <= 1 -> map through the loaded
word_index dict, OOV -> word_index['<unk>']).

This mirrors clean_medical_text()/EnhancedTokenizer.texts_to_sequences() in
chest_x_ray_data_processing/enhanced_data_loader.py, reimplemented inline
here (a) to avoid importing that module, which triggers nltk.download()
network calls at import time that can hang for minutes on a
network-restricted node, and (b) to pin down the exact fallback tokenizer
behavior (plain .split()) rather than whatever nltk.word_tokenize would do
if punkt happens to be available in the current environment.

Output: section_boundaries_test_paper2.csv
  columns: study_id, findings_token_count, has_findings, has_impression
"""

import os
import re
import csv
import pickle
import glob

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SHARD_TEST_DIR = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/test"
METADATA_PKL_PATH = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/metadata.pkl"
ORIGINAL_METADATA_CSV = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
REPORTS_DIR = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"

OUTPUT_CSV_PATH = os.path.join(PROJECT_DIR, "section_boundaries_test_paper2.csv")
MAX_SEQUENCE_LENGTH = 128  # matches metadata['max_sequence_length'] for mimic_shards_hybrid_full_ori

MEDICAL_KEEP_WORDS = {
    'no', 'not', 'normal', 'abnormal', 'present', 'absent',
    'mild', 'moderate', 'severe', 'large', 'small', 'right', 'left',
}


def load_word_index():
    """Load the saved word_index dict from metadata.pkl -- do NOT rebuild a new tokenizer."""
    with open(METADATA_PKL_PATH, 'rb') as f:
        metadata = pickle.load(f)
    word_index = metadata['tokenizer']
    assert isinstance(word_index, dict), f"Expected tokenizer to be a plain word_index dict, got {type(word_index)}"
    return word_index


def load_test_study_ids():
    """Read only the 'study_ids' field from every test shard (no images/captions)."""
    shard_paths = sorted(glob.glob(os.path.join(SHARD_TEST_DIR, "*.pkl")))
    study_ids = []
    for shard_path in shard_paths:
        with open(shard_path, 'rb') as f:
            shard_data = pickle.load(f)
        study_ids.extend(str(sid) for sid in shard_data['study_ids'])
    return study_ids


def extract_text_from_report(report_path):
    """Same logic as mimic_data_loader.extract_text_from_report(), inlined."""
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


def _load_stopwords():
    """
    Load nltk's English stopword list (already present locally at
    ~/nltk_data on this machine -- confirmed, no download() call needed/made).
    """
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    return stop_words - MEDICAL_KEEP_WORDS


def clean_medical_text_tokens(text, stop_words):
    """
    Reimplementation of clean_medical_text() from enhanced_data_loader.py,
    with the tokenization step fixed to plain .split() -- see module
    docstring for why this matches the actual production behavior rather
    than nltk.word_tokenize.
    """
    if text is None or text == "":
        return []

    text = text.lower()
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    text = re.sub(r'\bvs\.\b', 'versus', text)
    text = re.sub(r'\betc\.\b', 'etc', text)
    text = re.sub(r'\bdr\.\b', 'doctor', text)
    text = re.sub(r'\bpt\.\b', 'patient', text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return tokens


def texts_to_sequence(text, word_index, oov_index, stop_words):
    tokens = clean_medical_text_tokens(text, stop_words)
    return [word_index.get(tok, oov_index) for tok in tokens]


def main():
    print("Loading word_index from metadata.pkl (mimic_shards_hybrid_full_ori)...")
    word_index = load_word_index()
    oov_index = word_index.get('<unk>', 1)
    print(f"Vocabulary size: {len(word_index)}, OOV index: {oov_index}")

    print("Loading nltk English stopwords (already cached locally, no download)...")
    stop_words = _load_stopwords()
    print(f"Stopword set size (after excluding medical keep-words): {len(stop_words)}")

    print("\nReading study_ids directly from test shard files (no images loaded)...")
    study_ids = load_test_study_ids()
    n = len(study_ids)
    print(f"Test study_ids loaded: {n}")

    print("\nLoading original metadata CSV for report_file lookup...")
    meta_df = pd.read_csv(ORIGINAL_METADATA_CSV)
    meta_df = meta_df.set_index(meta_df['study_id'].astype(str))

    rows_out = []
    missing_in_csv = 0
    missing_report_file = 0

    for sid in study_ids:
        if sid not in meta_df.index:
            missing_in_csv += 1
            rows_out.append({
                'study_id': sid,
                'findings_token_count': 0,
                'has_findings': 0,
                'has_impression': 0,
            })
            continue

        report_file = meta_df.loc[sid, 'report_file']
        report_path = os.path.join(REPORTS_DIR, report_file)
        if not os.path.exists(report_path):
            missing_report_file += 1
            rows_out.append({
                'study_id': sid,
                'findings_token_count': 0,
                'has_findings': 0,
                'has_impression': 0,
            })
            continue

        findings, impression = extract_text_from_report(report_path)
        has_findings = 1 if findings.strip() != "" else 0
        has_impression = 1 if impression.strip() != "" else 0

        seq = texts_to_sequence(findings, word_index, oov_index, stop_words)
        findings_token_count = min(len(seq), MAX_SEQUENCE_LENGTH)

        rows_out.append({
            'study_id': sid,
            'findings_token_count': findings_token_count,
            'has_findings': has_findings,
            'has_impression': has_impression,
        })

    if missing_in_csv:
        print(f"Warning: {missing_in_csv} study_ids not found in {ORIGINAL_METADATA_CSV} (set to 0/0/0)")
    if missing_report_file:
        print(f"Warning: {missing_report_file} report files not found on disk (set to 0/0/0)")

    with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['study_id', 'findings_token_count', 'has_findings', 'has_impression'])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"\nSaved lookup file to: {OUTPUT_CSV_PATH}")
    print(f"File exists: {os.path.exists(OUTPUT_CSV_PATH)}")
    print(f"Rows written: {len(rows_out)}")

    print("\n=== First 5 rows ===")
    for r in rows_out[:5]:
        print(r)

    df_out = pd.DataFrame(rows_out)
    avg_count = df_out['findings_token_count'].mean()
    pct_no_findings = 100.0 * (df_out['has_findings'] == 0).mean()
    pct_no_impression = 100.0 * (df_out['has_impression'] == 0).mean()

    print("\n=== Summary stats ===")
    print(f"Average findings_token_count: {avg_count:.4f}")
    print(f"%% of study_ids with has_findings=0: {pct_no_findings:.2f}%")
    print(f"%% of study_ids with has_impression=0: {pct_no_impression:.2f}%")


if __name__ == "__main__":
    main()
