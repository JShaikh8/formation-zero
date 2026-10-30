"""The play index from dense camera readings, end to end on synthetic film."""

import numpy as np
import pandas as pd

from formation_zero.perception.play_index import build_from_dense

TAKE = 30                      # samples per take: 12 s at 0.4 s
FPS = 25.0
NULLIFIED = ("(Shotgun) M.Stafford pass incomplete short middle to P.Nacua.\n"
             "PENALTY on LAR-W.McClendon, Offensive Holding, 10 yards, enforced at LAR 30 - No Play.")
FALSE_START = "PENALTY on LAR-K.Dotson, False Start, 5 yards, enforced at SF 19 - No Play."


def synthetic(views: str, seed: int = 0):
    """Dense samples for a run of takes: the sideline camera bright on top, the end-zone camera below."""
    rng = np.random.default_rng(seed)
    n = TAKE * len(views)
    thumbs = np.zeros((n, 18, 32, 3), np.uint8)
    for k, v in enumerate(views):
        top, bottom = (190, 60) if v == "S" else (60, 190)
        block = np.full((TAKE, 18, 32, 3), bottom, np.int16)
        block[:, :9] = top
        thumbs[k * TAKE:(k + 1) * TAKE] = np.clip(block + rng.integers(-8, 9, block.shape), 0, 255)
    frames = np.arange(n) * 10
    dense = {"frames": frames, "diffs": np.ones(n, np.float32), "thumbs": thumbs,
             "orient": rng.random((n, 36)).astype(np.float32) * 0.01, "fps": FPS}
    shots = pd.DataFrame({"start_frame": frames[::TAKE]})
    return dense, shots


def pbp():
    types = ["kickoff", "pass", "no_play", "no_play", "run"]
    descs = ["a kickoff", "a pass", NULLIFIED, FALSE_START, "a run"]
    counted = [t != "no_play" for t in types]
    index = np.cumsum(counted) - 1
    return pd.DataFrame({
        "game_key": "g", "play_type": types, "desc": descs, "qtr": 1, "time": "10:00",
        "play_index": [float(i) if c else np.nan for i, c in zip(index, counted)],
        "play_uid": [f"g_p{i:03d}" if c else None for i, c in zip(index, counted)],
    })


def labels():
    # one labelled take per angle is enough to read this film
    return [{"start_s": 0.0, "end_s": TAKE * 10 / FPS, "view": "SIDELINE"},
            {"start_s": TAKE * 10 / FPS, "end_s": 2 * TAKE * 10 / FPS, "view": "ENDZONE"}]


def test_index_keeps_its_contract_and_reports_the_rest():
    # on film: kickoff, pass, the nullified pass, run; the false start was cut
    dense, shots = synthetic("SESESESE")
    index, report = build_from_dense(dense, shots, pbp(), labels())
    assert list(index["play_uid"]) == ["g_p000", "g_p001", "g_p002"]
    assert index["play_index"].notna().all()
    assert list(index["film_play_index"]) == [0, 1, 3]
    assert index["sideline_start_s"].iloc[1] == 2 * TAKE * 0.4
    assert [n["film_play_index"] for n in report["no_plays_on_film"]] == [2]
    assert report["optional"][0]["on_film"] is False
    assert report["skipped_slots"] == [] and report["junk"] == []


def test_both_angles_must_be_labelled():
    dense, shots = synthetic("SESE")
    try:
        build_from_dense(dense, shots, pbp(), labels()[:1])
    except ValueError as e:
        assert "both camera angles" in str(e)
    else:
        raise AssertionError("expected a ValueError")
