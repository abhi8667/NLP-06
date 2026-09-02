"""
Tests for Continuous Rényi Privacy Accounting across Federated Rounds (Phase P3B).
"""

import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from research_track.federation import ClinicalFlowerClient


class TestPrivacyAccounting(unittest.TestCase):

    def test_cumulative_epsilon_monotonic_growth(self):
        # 100 samples per client
        x = torch.randn(100, 12, 6, dtype=torch.float32)
        y = (torch.rand(100) < 0.2).float()
        loader = DataLoader(TensorDataset(x, y), batch_size=25, shuffle=True)

        client = ClinicalFlowerClient(
            site_id="site_test",
            train_loader=loader,
            test_loader=loader,
            architecture="DPLSTM",
            target_epsilon=2.0,
            total_rounds=4,
            local_epochs=1,
        )

        epsilons = []
        weights = client.get_parameters()

        for round_idx in range(4):
            weights, _, metrics = client.fit(weights)
            eps = metrics["achieved_epsilon"]
            epsilons.append(eps)

        # Confirm monotonic growth: round 1 < round 2 < round 3 < round 4
        self.assertEqual(len(epsilons), 4)
        for i in range(len(epsilons) - 1):
            self.assertLess(
                epsilons[i],
                epsilons[i + 1],
                f"Epsilon did not grow across rounds: {epsilons[i]} vs {epsilons[i+1]} (accountant was reset!)"
            )

        # Confirm final achieved epsilon is within reasonable range of target 2.0
        final_eps = epsilons[-1]
        self.assertAlmostEqual(final_eps, 2.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
