import unittest

from ftm.robust_mdp_exact import two_state_tv_example, worst_case_tv_row


class TVRobustMDPTests(unittest.TestCase):
    def test_worst_row_moves_mass_from_high_to_low_value(self):
        row, value = worst_case_tv_row((("good", 0.8), ("bad", 0.2)),
                                       {"good": 4.0, "bad": -2.0}, 0.15)
        self.assertAlmostEqual(row[0][1], 0.65)
        self.assertAlmostEqual(row[1][1], 0.35)
        self.assertAlmostEqual(value, 1.9)

    def test_value_iteration_returns_valid_policy_and_residual(self):
        result = two_state_tv_example().value_iteration(gamma=0.9)
        self.assertEqual(result.policy["decision"], "migrate")
        self.assertLessEqual(result.residual, 1e-10)
        self.assertGreaterEqual(result.iterations, 1)

    def test_rejects_invalid_probability_row(self):
        with self.assertRaises(ValueError):
            worst_case_tv_row((("a", 0.4), ("b", 0.4)), {"a": 1, "b": 0}, 0.2)


if __name__ == "__main__":
    unittest.main()
