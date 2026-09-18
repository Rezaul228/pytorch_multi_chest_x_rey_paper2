#!/usr/bin/env python3
"""
Unit test for the section-mechanism guards added to train_retrieval_v2_paper2.py
on 2026-09-18, after a ReXGradient MG-G2L run trained 100 epochs with
has_find/has_imp all-False and granularity loss exactly 0.000000 throughout.

Covers:
  assert_section_mechanism_active   -- raises iff every step of the epoch was 0.0
  assert_section_boundaries_loaded  -- raises on missing/empty/non-matching boundaries

No GPU, no data, no checkpoints: the helpers are pure functions over a tiny stub.
"""
import sys
import types
import unittest

# The guards live in their own dependency-free module, so this test needs no torch.
import section_guard_paper2 as T


class _StubDataset:
    def __init__(self, boundaries, study_ids):
        self.section_boundaries = boundaries
        self.samples = [{"study_ids": s} for s in study_ids]

    def __len__(self):
        return len(self.samples)


class TestGranularityGuard(unittest.TestCase):
    def test_raises_when_every_step_is_zero(self):
        with self.assertRaises(RuntimeError) as cm:
            T.assert_section_mechanism_active(epoch=41, epoch_gran_losses=[0.0] * 140)
        msg = str(cm.exception)
        self.assertIn("SECTION MECHANISM INERT", msg)
        self.assertIn("epoch 42", msg)            # 0-indexed in, 1-indexed out
        self.assertIn("section-boundary CSV", msg)  # names the likely cause

    def test_passes_when_any_step_is_non_zero(self):
        self.assertEqual(T.assert_section_mechanism_active(0, [0.0, 0.0, 0.31, 0.0]), 1)

    def test_passes_on_realistic_epoch(self):
        self.assertEqual(T.assert_section_mechanism_active(0, [0.42, 0.31, 0.28]), 3)

    def test_negative_zero_counts_as_zero(self):
        with self.assertRaises(RuntimeError):
            T.assert_section_mechanism_active(0, [-0.0, 0.0])


class TestBoundaryGuard(unittest.TestCase):
    def test_raises_when_no_attribute(self):
        obj = types.SimpleNamespace()
        with self.assertRaises(RuntimeError) as cm:
            T.assert_section_boundaries_loaded(obj, "train")
        self.assertIn("SECTION BOUNDARIES MISSING", str(cm.exception))

    def test_raises_when_empty(self):
        ds = _StubDataset({}, ["a", "b"])
        with self.assertRaises(FileNotFoundError) as cm:
            T.assert_section_boundaries_loaded(ds, "train")
        self.assertIn("SECTION BOUNDARIES EMPTY", str(cm.exception))

    def test_raises_when_ids_do_not_match(self):
        # the ReXGradient-style failure: a CSV exists but is keyed for another dataset
        ds = _StubDataset({"zzz": (5, True, True)}, ["a", "b", "c"])
        with self.assertRaises(RuntimeError) as cm:
            T.assert_section_boundaries_loaded(ds, "val")
        self.assertIn("DO NOT MATCH", str(cm.exception))

    def test_passes_when_ids_match(self):
        ds = _StubDataset({"a": (5, True, True), "b": (7, True, False)}, ["a", "b"])
        self.assertEqual(T.assert_section_boundaries_loaded(ds, "train"), 2)

    def test_passes_with_partial_match(self):
        # a few unknown study_ids are tolerated (the per-sample fallback handles them)
        ds = _StubDataset({"a": (5, True, True)}, ["a", "unknown"])
        self.assertEqual(T.assert_section_boundaries_loaded(ds, "train"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
