import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from yolo_lane_fit import fit_lanes


def result(polygons, classes, confidences):
    boxes = [SimpleNamespace(cls=np.array(cls), conf=np.array(conf))
             for cls, conf in zip(classes, confidences)]
    return SimpleNamespace(masks=SimpleNamespace(xy=polygons), boxes=boxes,
                           names={0: "left-sideline", 1: "right-sideline"})


class LaneFitTests(unittest.TestCase):
    def test_model_class_decides_side_and_coordinates_keep_source_size(self):
        # A vertical boundary must retain the model's side even with zero slope.
        polygon = np.array([[700, 200], [710, 200], [710, 900], [700, 900]])
        lines, mask = fit_lanes(result([polygon], [0], [0.9]), (960, 1280, 3))
        self.assertEqual(set(lines), {"left"})
        self.assertAlmostEqual(lines["left"]["top"][0], 705)
        self.assertEqual(mask.shape, (960, 1280))
        self.assertLessEqual(lines["left"]["reference_y_px"], 900)

    def test_missing_side_is_not_invented_and_highest_confidence_wins(self):
        a = np.array([[100, 100], [110, 100], [110, 470], [100, 470]])
        b = a + [100, 0]
        lines, mask = fit_lanes(result([a, b], [0, 0], [0.6, 0.95]), (480, 640, 3))
        self.assertEqual(set(lines), {"left"})
        self.assertAlmostEqual(lines["left"]["top"][0], 205)
        self.assertEqual(mask[200, 105], 0)
        self.assertEqual(mask[200, 205], 255)

    def test_horizontal_or_nonfinite_predictions_are_rejected(self):
        horizontal = np.array([[20, 200], [600, 200], [600, 205], [20, 205]])
        bad = horizontal.astype(float)
        bad[0, 0] = np.nan
        lines, mask = fit_lanes(result([horizontal, bad], [0, 1], [0.9, 0.9]), (480, 640, 3))
        self.assertEqual(lines, {})
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_empty_frame_result_clears_everything(self):
        lines, mask = fit_lanes(SimpleNamespace(masks=None), (480, 640, 3))
        self.assertEqual(lines, {})
        self.assertEqual(np.count_nonzero(mask), 0)


if __name__ == "__main__":
    unittest.main()
