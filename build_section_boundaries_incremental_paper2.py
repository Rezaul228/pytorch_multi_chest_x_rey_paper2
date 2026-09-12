#!/usr/bin/env python3
"""
Incremental, memory-flat section-boundary computation for MIMIC-CXR train/val
splits (mimic_shards_hybrid_full_ori), using the same verified tokenization
pipeline as build_section_boundaries_paper2.py (which already validated 100%
exact match against the actual stored caption_seq on a 300-sample random
check of the test split).

Text-only, CPU-only, no images, no model. Designed to run under a modest
fixed memory budget regardless of split size (155,800 train / 31,900 val
study_ids): shards are read and discarded ONE AT A TIME (only ~100 samples'
study_ids kept briefly, never their images/captions), and output rows are
written and flushed to disk immediately, one at a time -- nothing is
accumulated in a list or DataFrame before writing.

Output columns: study_id, findings_token_count, has_findings, has_impression
"""

import os
import re
import csv
import glob
import gc
import pickle
import argparse

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SHARD_BASE_DIR = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori"
METADATA_PKL_PATH = os.path.join(SHARD_BASE_DIR, "metadata.pkl")
ORIGINAL_METADATA_CSV = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
REPORTS_DIR = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"

MAX_SEQUENCE_LENGTH = 128

MEDICAL_KEEP_WORDS = {
    'no', 'not', 'normal', 'abnormal', 'present', 'absent',
    'mild', 'moderate', 'severe', 'large', 'small', 'right', 'left',
}


def load_word_index():
    with open(METADATA_PKL_PATH, 'rb') as f:
        metadata = pickle.load(f)
    word_index = metadata['tokenizer']
    assert isinstance(word_index, dict), f"Expected tokenizer to be a plain word_index dict, got {type(word_index)}"
    return word_index


def load_stopwords():
    from nltk.corpus import stopwords
    return set(stopwords.words('english')) - MEDICAL_KEEP_WORDS


def clean_medical_text_tokens(text, stop_words):
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

    del content
    return findings, impression


def process_split(split, output_csv_path):
    shard_dir = os.path.join(SHARD_BASE_DIR, split)
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.pkl")))
    print(f"[{split}] Found {len(shard_paths)} shard files in {shard_dir}")

    print(f"[{split}] Loading word_index from metadata.pkl...")
    word_index = load_word_index()
    oov_index = word_index.get('<unk>', 1)

    print(f"[{split}] Loading nltk English stopwords (cached locally)...")
    stop_words = load_stopwords()

    print(f"[{split}] Loading original metadata CSV for report_file lookup...")
    meta_df = pd.read_csv(ORIGINAL_METADATA_CSV)
    meta_df = meta_df.set_index(meta_df['study_id'].astype(str))

    total_written = 0
    missing_in_csv = 0
    missing_report_file = 0

    with open(output_csv_path, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['study_id', 'findings_token_count', 'has_findings', 'has_impression'])
        out_f.flush()

        for shard_idx, shard_path in enumerate(shard_paths):
            with open(shard_path, 'rb') as sf:
                shard_data = pickle.load(sf)
            # Only ever keep study_ids -- images/captions are dropped immediately.
            study_ids = [str(sid) for sid in shard_data['study_ids']]
            del shard_data
            gc.collect()

            for sid in study_ids:
                if sid not in meta_df.index:
                    missing_in_csv += 1
                    writer.writerow([sid, 0, 0, 0])
                    out_f.flush()
                    total_written += 1
                    continue

                report_file = meta_df.loc[sid, 'report_file']
                report_path = os.path.join(REPORTS_DIR, report_file)
                if not os.path.exists(report_path):
                    missing_report_file += 1
                    writer.writerow([sid, 0, 0, 0])
                    out_f.flush()
                    total_written += 1
                    continue

                findings, impression = extract_text_from_report(report_path)
                has_findings = 1 if findings.strip() != "" else 0
                has_impression = 1 if impression.strip() != "" else 0

                seq = texts_to_sequence(findings, word_index, oov_index, stop_words)
                findings_token_count = min(len(seq), MAX_SEQUENCE_LENGTH)

                writer.writerow([sid, findings_token_count, has_findings, has_impression])
                out_f.flush()
                total_written += 1

                # Discard this sample's text before moving to the next one.
                del findings, impression, seq

            if (shard_idx + 1) % 100 == 0 or (shard_idx + 1) == len(shard_paths):
                print(f"[{split}]   {shard_idx + 1}/{len(shard_paths)} shards processed, "
                      f"{total_written} rows written so far...")

    print(f"[{split}] DONE: {total_written} rows written to {output_csv_path}")
    if missing_in_csv:
        print(f"[{split}] Warning: {missing_in_csv} study_ids not found in {ORIGINAL_METADATA_CSV}")
    if missing_report_file:
        print(f"[{split}] Warning: {missing_report_file} report files not found on disk")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', required=True, choices=['train', 'val'])
    parser.add_argument('--output', required=True, help="Output CSV path")
    args = parser.parse_args()
    process_split(args.split, args.output)


if __name__ == "__main__":
    main()
