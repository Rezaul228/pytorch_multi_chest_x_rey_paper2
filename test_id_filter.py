#!/usr/bin/env python3
"""
Unit tests for the optional --train_ids/--val_ids study_id filter (scale ablation).
Uses tiny synthetic shards; no GPU, no real data.
"""
import os
import pickle
import tempfile
import unittest

import numpy as np

import id_filter
import data_loader_v1 as L1
import data_loader_v1_paper2 as L2


def _make_shards(root, n_shards=3, per=5):
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    paths_, all_ids = [], []
    k = 0
    for s in range(n_shards):
        ids = np.array([str(1000 + k + i) for i in range(per)])
        d = {"images": np.random.RandomState(s).rand(per, 4, 4, 3).astype(np.float32),
             "captions": np.arange(per * 8, dtype=np.int32).reshape(per, 8) + s,
             "study_ids": ids}
        p = os.path.join(root, "train", f"shard_{s:04d}.pkl")
        with open(p, "wb") as f:
            pickle.dump(d, f)
        paths_.append(p); all_ids += list(ids); k += per
    return paths_, all_ids


class TestHelper(unittest.TestCase):
    def _csv(self, rows, header="study_id"):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        f.write(header + "\n" + "\n".join(rows) + ("\n" if rows else "")); f.close()
        return f.name

    def test_loads_ids_as_str_set(self):
        self.assertEqual(id_filter.load_id_filter(self._csv(["5", "12", "007"])), frozenset({"5", "12", "007"}))

    def test_rejects_duplicates_empty_and_bad_header(self):
        with self.assertRaises(ValueError):
            id_filter.load_id_filter(self._csv(["5", "5"]))
        with self.assertRaises(ValueError):
            id_filter.load_id_filter(self._csv([]))
        with self.assertRaises(ValueError):
            id_filter.load_id_filter(self._csv(["5"], header="id"))

    def test_keep_sample(self):
        self.assertTrue(id_filter.keep_sample(5, None))            # no filter -> keep all
        self.assertTrue(id_filter.keep_sample(5, frozenset({"5"})))  # int id matches str key
        self.assertFalse(id_filter.keep_sample(6, frozenset({"5"})))

    def test_assert_filter_satisfied(self):
        id_filter.assert_filter_satisfied(3, frozenset({"a", "b", "c"}), "train")
        id_filter.assert_filter_satisfied(999, None, "train")       # filter off -> never raises
        with self.assertRaises(RuntimeError):
            id_filter.assert_filter_satisfied(2, frozenset({"a", "b", "c"}), "train")


class TestDatasetFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.shards, self.all_ids = _make_shards(self.tmp)

    def _datasets(self, **kw):
        return [L1.IndianaDataset(self.shards, None, **kw),
                L2.IndianaDataset(self.shards, None, split_name="train", **kw)]

    def test_filter_keeps_exactly_the_listed_ids(self):
        want = frozenset({self.all_ids[0], self.all_ids[7], self.all_ids[14]})   # one per shard
        for ds in self._datasets(keep_ids=want):
            got = [str(s["study_ids"]) for s in ds.samples]
            self.assertEqual(sorted(got), sorted(want))
            self.assertEqual(len(ds), 3)

    def test_filter_off_is_the_original_behaviour(self):
        for ds in self._datasets():
            self.assertEqual([str(s["study_ids"]) for s in ds.samples], self.all_ids)   # everything, shard order
        for ds in self._datasets(keep_ids=None):
            self.assertEqual(len(ds), len(self.all_ids))

    def test_filtered_content_matches_unfiltered_content(self):
        want = frozenset({self.all_ids[3], self.all_ids[11]})
        full = {str(s["study_ids"]): s for s in L1.IndianaDataset(self.shards, None).samples}
        for ds in self._datasets(keep_ids=want):
            for s in ds.samples:
                ref = full[str(s["study_ids"])]
                self.assertTrue(np.array_equal(s["images"], ref["images"]))
                self.assertTrue(np.array_equal(s["captions"], ref["captions"]))

    def test_kept_samples_are_copies_not_views(self):
        # a view (.base is the shard array) would pin every touched shard in memory
        for ds in self._datasets(keep_ids=frozenset({self.all_ids[0]})):
            self.assertIsNone(ds.samples[0]["images"].base)

    def test_max_samples_still_counts_kept_samples(self):
        want = frozenset(self.all_ids[:10])
        ds = L1.IndianaDataset(self.shards, 4, keep_ids=want)
        self.assertEqual(len(ds), 4)

    def test_unknown_ids_are_detected_by_the_trainer_check(self):
        want = frozenset({self.all_ids[0], "does_not_exist"})
        ds = L1.IndianaDataset(self.shards, None, keep_ids=want)
        with self.assertRaises(RuntimeError):
            id_filter.assert_filter_satisfied(len(ds), want, "train")


if __name__ == "__main__":
    unittest.main(verbosity=2)
