"""
Tests for Data Pipeline, 4-6h Horizon Windowing, and Patient Split Isolation (Phase P2).
"""

import unittest
from pathlib import Path
import numpy as np

from research_track.data import DatasetBuilder, load_split_arrays


class TestDataPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processed_dir = Path("research_track/data/processed")
        cls.builder = DatasetBuilder(output_dir=cls.processed_dir)
        cls.manifest = cls.builder.build_dataset(max_patients_per_site=50)

    def test_window_shapes_and_types(self):
        w_tr, y_tr, _ = load_split_arrays(self.processed_dir, "site_a", "train")
        self.assertGreater(len(w_tr), 0)
        self.assertEqual(w_tr.ndim, 3)
        self.assertEqual(w_tr.shape[1], 12)  # 12 timesteps
        self.assertEqual(w_tr.shape[2], 6)   # 6 vitals
        self.assertEqual(w_tr.dtype, np.float32)

        self.assertEqual(len(w_tr), len(y_tr))
        self.assertEqual(y_tr.dtype, np.float32)
        self.assertTrue(set(np.unique(y_tr)).issubset({0.0, 1.0}))

    def test_zero_nans_in_windows(self):
        for site_id in ["site_a", "site_b"]:
            for split_type in ["train", "test"]:
                w, _, _ = load_split_arrays(self.processed_dir, site_id, split_type)
                self.assertEqual(np.isnan(w).sum(), 0, f"Found NaNs in {site_id}_{split_type}")

    def test_strict_patient_split_isolation(self):
        for site_id in ["site_a", "site_b"]:
            _, _, tr_pids = load_split_arrays(self.processed_dir, site_id, "train")
            _, _, te_pids = load_split_arrays(self.processed_dir, site_id, "test")

            overlap = set(tr_pids).intersection(set(te_pids))
            self.assertEqual(len(overlap), 0, f"Patient leakage detected between train and test in {site_id}: {overlap}")

    def test_manifest_and_sha256_hashes(self):
        manifest_path = self.processed_dir / "freeze_manifest.json"
        self.assertTrue(manifest_path.exists())
        self.assertIn("file_hashes", self.manifest)
        self.assertIn("imputation_rates", self.manifest)
        self.assertIn("HR", self.manifest["imputation_rates"])
        self.assertIn("Glucose", self.manifest["imputation_rates"])


if __name__ == "__main__":
    unittest.main()
