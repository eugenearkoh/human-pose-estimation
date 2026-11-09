import math
import unittest

from src.pose_estimator import calculate_angle


class CalculateAngleTests(unittest.TestCase):
    def test_right_angle(self):
        self.assertAlmostEqual(
            calculate_angle((1, 0), (0, 0), (0, 1)), 90.0, places=6
        )

    def test_straight_angle(self):
        self.assertAlmostEqual(
            calculate_angle((-1, 0), (0, 0), (1, 0)), 180.0, places=6
        )

    def test_zero_length_segment(self):
        self.assertTrue(math.isnan(calculate_angle((0, 0), (0, 0), (1, 1))))


if __name__ == "__main__":
    unittest.main()
