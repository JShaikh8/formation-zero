"""Suggested removals from the field mask: feet outside it, judged at the bottom middle of the box."""

import numpy as np

from formation_zero.configs.nfl_field import FIELD_WIDTH_YD
from formation_zero.labelling.field_filter import feet_off, foot_outside
from formation_zero.perception.field import homography as hg


def test_feet_off_the_field_are_suggested_and_feet_past_the_frame_are_not():
    mask = np.zeros((100, 100), bool)
    mask[40:, :] = True                                   # the field is the lower part of the frame
    boxes = [{"box": [10, 50, 20, 90]},                   # on the field
             {"box": [30, 5, 40, 30]},                    # bench, above the far sideline
             {"box": [50, 70, 60, 130]}]                  # feet below the frame: cannot tell, keep
    assert foot_outside(boxes, mask) == [1]


def test_in_yards_feet_past_either_sideline_are_off_and_feet_on_the_field_are_not():
    c = np.array([60.0, -87.3, 27.2])                     # the press-box camera, looking across the field
    fwd = np.array([60.0, 27.0, 0.0]) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    r = np.vstack([right, np.cross(fwd, right), fwd])
    k = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    p = k @ np.hstack([r, -(r @ c)[:, None]])
    h = p[:, [0, 1, 3]]
    g = np.linalg.inv(h)

    def box(y):                                           # a 2-yd person standing at field (60, y), feet in the box's bottom
        head, foot = p @ [60, y, 2, 1], p @ [60, y, 0, 1]
        x0, y1, y2 = foot[0] / foot[2], head[1] / head[2], foot[1] / foot[2]
        return {"box": [x0 - 15, y1, x0 + 15, y2]}

    boxes = [box(20.0), box(-3.0), box(FIELD_WIDTH_YD + 3.0), box(0.2), box(FIELD_WIDTH_YD - 0.2)]
    assert np.allclose(hg.apply(g, np.array([[boxes[0]["box"][0] + 15, boxes[0]["box"][3]]]))[0], [60, 20], atol=1e-6)
    assert feet_off(boxes, (1080, 1920), g, c) == [1, 2]
