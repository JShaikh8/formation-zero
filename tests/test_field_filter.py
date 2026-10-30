"""Suggested removals from the field mask: feet outside it, judged at the bottom middle of the box."""

import numpy as np

from formation_zero.labelling.field_filter import foot_outside


def test_feet_off_the_field_are_suggested_and_feet_past_the_frame_are_not():
    mask = np.zeros((100, 100), bool)
    mask[40:, :] = True                                   # the field is the lower part of the frame
    boxes = [{"box": [10, 50, 20, 90]},                   # on the field
             {"box": [30, 5, 40, 30]},                    # bench, above the far sideline
             {"box": [50, 70, 60, 130]}]                  # feet below the frame: cannot tell, keep
    assert foot_outside(boxes, mask) == [1]
