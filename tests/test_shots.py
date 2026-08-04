"""Tests for segmenting continuous film into shots.

Shot boundaries are the spine of the play index: a missed cut merges two camera angles into one
segment, and a spurious cut splits a play in half. Both corrupt every label derived downstream,
so the grouping rules are pinned here. Decoding is not exercised — `group` takes plain samples
precisely so the logic is testable without a video file or the [cv] extra installed.
"""

import numpy as np
import pytest

from gridiron.perception import shots
from gridiron.perception.shots import Shot, group, line_tilt, to_frame


def samples(*runs, step=0.4, cut_diff=40.0, tilt=80.0):
    """Build a sample stream from run lengths, inserting a cut spike before each run but the first.

    ``samples(3, 2)`` -> 3 samples, a cut, then 2 samples.
    """
    out, time = [], 0.0
    for index, length in enumerate(runs):
        for position in range(length):
            difference = cut_diff if (index and position == 0) else 1.0
            out.append((round(time, 3), difference, tilt))
            time += step
    return out


# ---- grouping ------------------------------------------------------------------------------


def test_group_splits_at_difference_spikes():
    result = group(samples(30, 40, 35))
    assert len(result) == 3
    assert [s.samples for s in result] == [30, 40, 35]


def test_group_duration_spans_the_last_sample():
    """A run of n samples covers n*step seconds, not (n-1)*step — the last sample has extent."""
    result = group(samples(10), step=0.4)
    assert result[0].duration == 4.0


def test_group_returns_one_shot_when_nothing_exceeds_the_threshold():
    assert len(group(samples(50))) == 1


def test_group_handles_empty_input():
    assert group([]) == []


def test_group_threshold_is_respected():
    """A difference just under the threshold is camera motion, not a cut."""
    stream = samples(20, 20, cut_diff=shots.CUT_THRESHOLD - 0.1)
    assert len(group(stream)) == 1


# ---- fragment absorption -------------------------------------------------------------------


def test_short_fragments_are_absorbed_into_the_preceding_shot():
    """Camera whip mid-take reads as a cut; the resulting sliver is not its own shot."""
    result = group(samples(40, 2, 40))          # 2 samples = 0.8s, under MIN_DURATION
    assert len(result) == 2
    assert result[0].samples == 42              # the sliver's samples join the first shot


def test_absorbed_fragment_extends_the_previous_shot_to_meet_the_next():
    result = group(samples(40, 2, 40))
    assert result[0].end == pytest.approx(result[1].start)


def test_leading_fragment_survives_when_there_is_nothing_to_absorb_it_into():
    """Nothing precedes the first shot, so a short opening segment is kept rather than dropped."""
    result = group(samples(2, 40))
    assert len(result) == 2
    assert result[0].samples == 2


def test_consecutive_fragments_all_fold_into_one_shot():
    result = group(samples(40, 2, 2, 2, 40))
    assert len(result) == 2


# ---- tilt ----------------------------------------------------------------------------------


def grating(rows=168, cols=300, period=20.0, di=0.0, dj=0.0):
    """Sinusoidal grating whose bands are perpendicular to the direction (di, dj).

    Smooth rather than hard-edged: a hard stripe pattern on a discrete grid aliases into dotted
    steps along the diagonal, which is a property of the sampling and not of the orientation.
    """
    i, j = np.mgrid[0:rows, 0:cols]
    return (127.0 + 127.0 * np.sin(2 * np.pi * (di * i + dj * j) / period)).astype(np.float32)


def test_tilt_reads_0_for_horizontal_lines():
    """Bands running across the image — intensity varies down the rows only."""
    assert line_tilt(grating(di=1.0)) < 5.0


def test_tilt_reads_90_for_vertical_lines():
    assert line_tilt(grating(dj=1.0)) > 85.0


def test_tilt_reads_45_for_diagonal_lines():
    assert 40.0 < line_tilt(grating(di=1.0, dj=1.0)) < 50.0


def test_tilt_is_nan_on_a_featureless_frame():
    """A flat frame has no lines to measure; callers must not read a number into that."""
    assert np.isnan(line_tilt(np.full((168, 300), 128.0, dtype=np.float32)))


def test_shot_tilt_is_none_when_every_sample_is_featureless():
    stream = [(round(i * 0.4, 3), 1.0, float("nan")) for i in range(20)]
    assert group(stream)[0].tilt is None


def test_shot_tilt_ignores_the_frames_straddling_the_cut():
    """Interior frames decide the tilt, so a neighbouring angle bleeding in cannot capture it."""
    stream = [(round(i * 0.4, 3), 1.0, 10.0 if i < 3 or i > 16 else 85.0) for i in range(20)]
    assert group(stream)[0].tilt == 85.0


# ---- output shape --------------------------------------------------------------------------


def test_to_frame_carries_frame_numbers_for_the_aligned_renditions():
    """Renditions share a timebase, so a frame index found on the proxy addresses the 1080p file."""
    table = to_frame([Shot(start=100.0, duration=14.0, tilt=85.0, samples=35)], "gk", fps=25.0)
    row = table.iloc[0]
    assert (row["start_frame"], row["end_frame"]) == (2500, 2850)


def test_to_frame_is_ordered_and_indexed():
    table = to_frame([Shot(10.0, 5.0, 80.0, 12), Shot(15.0, 6.0, 40.0, 15)], "gk", fps=25.0)
    assert list(table["shot_index"]) == [0, 1]
    assert list(table["start_s"]) == [10.0, 15.0]


def test_to_frame_on_no_shots_produces_an_empty_table():
    assert to_frame([], "gk", fps=25.0).empty
