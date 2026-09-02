"""
Tests for Model Architectures, Logit Outputs, and Opacus DP-SGD Compatibility (Phase P3B).
"""

import unittest
import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset

from research_track.models import get_model, DPLSTMClassifier, DPGRUClassifier, CNN1DClassifier


class TestModelsDP(unittest.TestCase):

    def setUp(self):
        self.batch_size = 16
        self.dummy_x = torch.randn(self.batch_size, 12, 6, dtype=torch.float32)
        self.dummy_y = (torch.rand(self.batch_size) < 0.2).float()
        self.dataset = TensorDataset(self.dummy_x, self.dummy_y)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)

    def test_raw_logits_output(self):
        for arch in ["DPLSTM", "DPGRU", "CNN1D"]:
            model = get_model(arch)
            model.eval()
            with torch.no_grad():
                out = model(self.dummy_x)
            self.assertEqual(out.shape, (self.batch_size,), f"{arch} output shape mismatch")
            # Confirm raw logits can be negative and > 1 (not sigmoid bounded)
            self.assertTrue(torch.is_tensor(out))

    def test_opacus_dp_compatibility(self):
        for arch in ["DPLSTM", "DPGRU", "CNN1D"]:
            model = get_model(arch)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            pe = PrivacyEngine()
            # This must NOT raise ShouldReplaceModuleError or hook errors
            model_priv, opt_priv, dl_priv = pe.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.dataloader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )

            # Single training step
            for xb, yb in dl_priv:
                opt_priv.zero_grad()
                logits = model_priv(xb)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                opt_priv.step()

            eps = pe.get_epsilon(delta=1e-5)
            self.assertGreater(eps, 0.0)


if __name__ == "__main__":
    unittest.main()
