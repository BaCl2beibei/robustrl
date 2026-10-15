"""Joint action probabilities and auxiliary labels for the A/B/C ablation."""

import unittest
from importlib.util import find_spec


@unittest.skipUnless(find_spec("torch"), "PyTorch learning extra is not installed")
class HierarchicalPPOTests(unittest.TestCase):
    def test_joint_probability_and_all_targets_masked(self):
        import torch
        from ftm.hierarchical_ppo import build_hierarchical_actor_critic

        model = build_hierarchical_actor_critic(8, 4)
        state = torch.randn(2, 8)
        mask = torch.tensor([[True, False, True, True],
                             [False, False, False, True]])
        logits, values = model(state, mask)
        probabilities = torch.softmax(logits, dim=-1)
        gate_probability = torch.sigmoid(model.gate_logits(state[0:1]))[0]
        self.assertEqual(tuple(values.shape), (2,))
        self.assertAlmostEqual(float(probabilities[0].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(probabilities[0, :3].sum()),
                               float(gate_probability), places=6)
        self.assertEqual(float(probabilities[0, 1]), 0.0)
        self.assertAlmostEqual(float(probabilities[1, -1]), 1.0, places=6)

    def test_gate_labels_use_clear_time_margin_and_unrecoverable_flag(self):
        import math
        import torch
        from ftm.hierarchical_ppo import gate_labels

        state = torch.zeros(3, 16)
        masks = torch.tensor([[True, False, True], [True, False, True],
                              [True, False, True]])
        state[0, 2] = math.asinh(0.2)
        state[0, 6] = math.asinh(0.5)
        state[0, 7] = 1
        state[1, 2] = math.asinh(0.5)
        state[1, 6] = math.asinh(0.2)
        state[1, 7] = 1
        state[2, 2] = math.asinh(0.5)
        labels, selected = gate_labels(state, masks, 2)
        self.assertEqual(labels.tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(selected.tolist(), [True, True, True])


if __name__ == "__main__":
    unittest.main()
