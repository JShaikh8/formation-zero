"""Take alignment on synthetic film: each defect the clear game showed, one at a time."""

import numpy as np

from formation_zero.perception.align import ENDZONE, JUNK, SIDELINE, align, missed_cut_candidates

TAKE = 30          # samples per take: 12 s at 0.4 s


def film(views, take=TAKE, p_right=0.85):
    """Angle readings for a run of takes, and the sample indices where each take starts."""
    p = np.concatenate([np.full(take, p_right if v == "S" else 1 - p_right) for v in views])
    return p, [i * take for i in range(1, len(views))]


def takes(result):
    return [(s.start, s.end, s.view) for s in result.segments]


def test_clean_film_fills_every_slot_in_order():
    p, cuts = film("SESESE")
    r = align(p, cuts, [], 3)
    assert [s.view for s in r.segments] == [SIDELINE, ENDZONE] * 3
    assert [s.slot for s in r.segments] == [1, 2, 3, 4, 5, 6]
    assert r.merged_cuts == [] and r.skipped_slots == []


def test_a_false_cut_inside_a_take_is_merged():
    p, cuts = film("SESESE")
    cuts = sorted(cuts + [TAKE + 12])         # a fragment boundary inside the first end-zone take
    r = align(p, cuts, [], 3)
    assert r.merged_cuts == [TAKE + 12]
    assert takes(r)[1] == (TAKE, 2 * TAKE, ENDZONE)


def test_a_missed_cut_is_recovered_from_a_weaker_jump():
    p, cuts = film("SESESE")
    cuts.remove(3 * TAKE)                     # the cut between play 2's takes went undetected
    r = align(p, cuts, [3 * TAKE, 3 * TAKE + 9], 3)
    assert r.missed_cuts == [3 * TAKE]
    assert takes(r)[2:4] == [(2 * TAKE, 3 * TAKE, SIDELINE), (3 * TAKE, 4 * TAKE, ENDZONE)]


def test_pregame_from_one_camera_is_left_out():
    """A minute of pregame shot by the sideline camera, then three plays."""
    pre = 150
    p, cuts = film("SESESE")
    p = np.concatenate([np.full(pre, 0.85), p])
    cuts = [pre] + [c + pre for c in cuts] + [60]          # a cut inside the pregame too
    r = align(p, sorted(cuts), [], 3)
    plays = [s for s in r.segments if s.view != JUNK]
    assert all(s.view == JUNK for s in r.segments if s.end <= pre)
    assert plays[0].start == pre and [s.slot for s in plays] == [1, 2, 3, 4, 5, 6]


def test_the_count_holds_even_when_one_take_reads_wrong():
    """A take the classifier gets backwards cannot shift the pairing: alternation and cuts hold it."""
    p, cuts = film("SESESE")
    p[2 * TAKE:3 * TAKE] = 0.2                 # play 2's sideline take reads as end zone
    r = align(p, cuts, [], 3)
    assert [s.slot for s in r.segments] == [1, 2, 3, 4, 5, 6]
    assert r.merged_cuts == []


def test_missed_cut_candidates_are_spikes_not_already_cuts():
    d = np.ones(40)
    d[10], d[20], d[30] = 9.0, 12.0, 2.5
    assert missed_cut_candidates(d, exclude={20}) == [10]


def test_an_optional_play_the_film_lacks_is_skipped_cheaply():
    """Four plays on the list, the second a pre-snap foul the film dropped: three plays of takes."""
    p, cuts = film("SESESE")
    r = align(p, cuts, [], 4, optional=[False, True, False, False])
    assert r.skipped_slots == [3, 4]
    assert [s.slot for s in r.segments] == [1, 2, 5, 6, 7, 8]


def test_without_the_flag_a_missing_play_costs_a_lot():
    p, cuts = film("SESESE")
    cheap = align(p, cuts, [], 4, optional=[False, True, False, False]).score
    dear = align(p, cuts, [], 4).score
    assert dear < cheap - 10
