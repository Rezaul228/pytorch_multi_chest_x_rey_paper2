#!/usr/bin/env python3
"""
Validate the reverse-engineered tokenization pipeline (from
build_section_boundaries_paper2.py) against the ACTUAL stored caption_seq
in the real test shards, on a random sample of 300 study_ids out of the
12,429-sample MIMIC-CXR test set.

Read-only: does not modify any model file, does not modify
section_boundaries_test_paper2.csv, does not use IndianaDataLoader (avoids
loading images -- reads only 'captions' and 'study_ids' fields directly out
of each test shard pickle).

For each sampled study_id:
  - reconstructs the FULL combined-text (findings + impression) token
    sequence using the exact same pipeline as build_section_boundaries_paper2.py
  - truncates it to MAX_SEQUENCE_LENGTH (matching truncating='post' used
    when shards were built)
  - compares it, token-for-token, against the actual stored caption_seq for
    that study_id, up to where padding (value 0) starts in the actual sequence
  - reports exact-match rate, and for mismatches, whether the first
    differing position falls before or after that study_id's
    findings_token_count boundary (from section_boundaries_test_paper2.csv)
"""

import os
import re
import glob
import pickle
import random

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SHARD_TEST_DIR = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/test"
METADATA_PKL_PATH = "/home/abedin/Developments/chest_x_ray_data_processing/all_processed_data/mimic_shards_hybrid_full_ori/metadata.pkl"
ORIGINAL_METADATA_CSV = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/metadata/processed_metadata_hybrid.csv"
REPORTS_DIR = "/home/abedin/Developments/mimic_cxr-raw-data/mimic-cxr/organized_data/reports"
SECTION_BOUNDARIES_CSV = os.path.join(PROJECT_DIR, "section_boundaries_test_paper2.csv")

MAX_SEQUENCE_LENGTH = 128
SAMPLE_SIZE = 300
RANDOM_SEED = 42  # fixed for reproducibility; not specified by the task

MEDICAL_KEEP_WORDS = {
    'no', 'not', 'normal', 'abnormal', 'present', 'absent',
    'mild', 'moderate', 'severe', 'large', 'small', 'right', 'left',
}


def load_word_index():
    with open(METADATA_PKL_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return metadata['tokenizer']


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
    return findings, impression


def combine_findings_impression(findings, impression):
    """Same convention as mimic_data_loader.py's process_study_entries()."""
    combined_text = ""
    if findings:
        combined_text += findings
    if impression:
        if combined_text:
            combined_text += " " + impression
        else:
            combined_text = impression
    return combined_text


def load_actual_captions_and_study_ids():
    """Read only 'captions' and 'study_ids' from every test shard (no images)."""
    shard_paths = sorted(glob.glob(os.path.join(SHARD_TEST_DIR, "*.pkl")))
    study_id_to_caption = {}
    for shard_path in shard_paths:
        with open(shard_path, 'rb') as f:
            shard_data = pickle.load(f)
        for sid, cap in zip(shard_data['study_ids'], shard_data['captions']):
            study_id_to_caption[str(sid)] = np.asarray(cap)
    return study_id_to_caption


def actual_real_length(caption_seq, pad_value=0):
    """Index of first pad token (post-padding convention); len(seq) if none found."""
    nz = np.where(caption_seq == pad_value)[0]
    return int(nz[0]) if len(nz) > 0 else len(caption_seq)


def main():
    print("Loading word_index from metadata.pkl...")
    word_index = load_word_index()
    oov_index = word_index.get('<unk>', 1)

    print("Loading nltk stopwords (cached locally)...")
    stop_words = load_stopwords()

    print("Loading actual captions + study_ids directly from test shards (no images)...")
    study_id_to_caption = load_actual_captions_and_study_ids()
    print(f"Loaded {len(study_id_to_caption)} study_id -> caption_seq entries")

    print("Loading section_boundaries_test_paper2.csv (read-only, for boundary comparison)...")
    boundaries_df = pd.read_csv(SECTION_BOUNDARIES_CSV, dtype={'study_id': str})
    boundaries_df = boundaries_df.set_index('study_id')

    print("Loading original metadata CSV for report_file lookup...")
    meta_df = pd.read_csv(ORIGINAL_METADATA_CSV)
    meta_df = meta_df.set_index(meta_df['study_id'].astype(str))

    all_study_ids = list(study_id_to_caption.keys())
    random.seed(RANDOM_SEED)
    sample_ids = random.sample(all_study_ids, SAMPLE_SIZE)
    print(f"\nSampled {len(sample_ids)} study_ids (seed={RANDOM_SEED}) out of {len(all_study_ids)}")

    exact_matches = 0
    mismatches = []

    for sid in sample_ids:
        actual_seq = study_id_to_caption[sid]
        actual_len = actual_real_length(actual_seq)
        actual_prefix = actual_seq[:actual_len].tolist()

        if sid not in meta_df.index:
            mismatches.append({
                'study_id': sid, 'reason': 'study_id not found in original metadata CSV',
                'actual': actual_prefix, 'reconstructed': None, 'first_diff_pos': None,
            })
            continue

        report_file = meta_df.loc[sid, 'report_file']
        report_path = os.path.join(REPORTS_DIR, report_file)
        findings, impression = extract_text_from_report(report_path)
        combined_text = combine_findings_impression(findings, impression)

        reconstructed_full = texts_to_sequence(combined_text, word_index, oov_index, stop_words)
        reconstructed_trunc = reconstructed_full[:MAX_SEQUENCE_LENGTH]

        if reconstructed_trunc == actual_prefix:
            exact_matches += 1
        else:
            first_diff_pos = None
            min_len = min(len(reconstructed_trunc), len(actual_prefix))
            for i in range(min_len):
                if reconstructed_trunc[i] != actual_prefix[i]:
                    first_diff_pos = i
                    break
            if first_diff_pos is None:
                first_diff_pos = min_len  # one is a strict prefix of the other -> lengths differ

            findings_boundary = None
            if sid in boundaries_df.index:
                findings_boundary = int(boundaries_df.loc[sid, 'findings_token_count'])

            side = None
            if findings_boundary is not None:
                side = "BEFORE" if first_diff_pos < findings_boundary else "AFTER"

            mismatches.append({
                'study_id': sid,
                'reason': 'token mismatch',
                'actual': actual_prefix,
                'reconstructed': reconstructed_trunc,
                'first_diff_pos': first_diff_pos,
                'findings_token_count_boundary': findings_boundary,
                'mismatch_relative_to_boundary': side,
            })

    n = len(sample_ids)
    match_rate = 100.0 * exact_matches / n

    print("\n" + "=" * 70)
    print("EXACT-MATCH RESULT")
    print("=" * 70)
    print(f"Exact matches: {exact_matches} / {n}  ({match_rate:.2f}%)")
    print(f"Mismatches: {len(mismatches)} / {n}")

    if mismatches:
        before_count = sum(1 for m in mismatches if m.get('mismatch_relative_to_boundary') == 'BEFORE')
        after_count = sum(1 for m in mismatches if m.get('mismatch_relative_to_boundary') == 'AFTER')
        unknown_count = len(mismatches) - before_count - after_count
        print(f"\nOf the {len(mismatches)} mismatches:")
        print(f"  First difference BEFORE the findings_token_count boundary: {before_count}")
        print(f"  First difference AFTER  the findings_token_count boundary: {after_count}")
        if unknown_count:
            print(f"  Boundary unknown / other reason: {unknown_count}")

        print("\n" + "=" * 70)
        print("MISMATCH DETAILS (full)")
        print("=" * 70)
        for m in mismatches:
            print(f"\nstudy_id={m['study_id']}  reason={m['reason']}")
            if m['reason'] == 'token mismatch':
                print(f"  findings_token_count boundary: {m['findings_token_count_boundary']}")
                print(f"  first_diff_pos: {m['first_diff_pos']}  ({m['mismatch_relative_to_boundary']} the boundary)")
                print(f"  ACTUAL        ({len(m['actual'])} tokens): {m['actual']}")
                print(f"  RECONSTRUCTED ({len(m['reconstructed'])} tokens): {m['reconstructed']}")
            else:
                print(f"  ACTUAL ({len(m['actual'])} tokens): {m['actual']}")
    else:
        print("\nNo mismatches -- all sampled study_ids matched exactly.")


if __name__ == "__main__":
    main()
