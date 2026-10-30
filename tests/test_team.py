"""Team split by uniform colour, on made-up crops: a player is a shirt band over a pants band on a field."""

import numpy as np
import pytest

from formation_zero.perception.team import MIN_HEIGHT, PALETTE, ColourTeams, crop_features

GREEN, SNOW = (70, 110, 60), (200, 205, 210)
WHITE, ROYAL, NAVY = (240, 240, 245), (40, 70, 200), (25, 30, 60)
RAMS, BEARS = (WHITE, ROYAL), (NAVY, WHITE)  # (shirt, pants)


def player(kit, field, h=90, w=40, rng=None, noise=8.0):
    """A field-coloured box with a body in the middle columns: shirt on top, pants below."""
    rng = rng or np.random.default_rng(0)
    img = np.empty((h, w, 3), np.float32)
    img[:] = field
    x0, x1 = int(w * (0.3 + rng.uniform(-0.05, 0.05))), int(w * (0.7 + rng.uniform(-0.05, 0.05)))
    img[int(h * 0.1):int(h * 0.48), x0:x1] = kit[0]
    img[int(h * 0.48):int(h * 0.85), x0:x1] = kit[1]
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def game(n=40, seed=0):
    """Both teams on both fields, assorted sizes; returns crops and true team (0 = Rams, the white shirts)."""
    rng = np.random.default_rng(seed)
    crops, truth = [], []
    for i in range(n):
        team, field = i % 2, (GREEN, SNOW)[(i // 2) % 2]
        h = int(rng.integers(30, 160))
        crops.append(player((RAMS, BEARS)[team], field, h=h, w=max(8, h * 2 // 5), rng=rng))
        truth.append(team)
    return crops, np.array(truth)


def test_features_are_two_unit_histograms():
    f = crop_features(player(RAMS, GREEN))
    n = len(PALETTE)
    assert f.shape == (2 * n,)
    assert np.isclose((f[:n] ** 2).sum(), 1) and np.isclose((f[n:] ** 2).sum(), 1)


def test_unusable_crops_have_no_features():
    assert crop_features(player(RAMS, GREEN, h=MIN_HEIGHT - 1, w=12)) is None
    assert crop_features(np.zeros((60, 30), np.uint8)) is None  # not RGB
    assert crop_features(player(RAMS, GREEN, h=MIN_HEIGHT, w=12)) is not None


def test_the_field_hardly_matters():
    """A white shirt on snow still looks like a white shirt on grass. Without the background weighting
    the two are 0.8 of the way to the other team; with it, 0.3."""
    on_grass, on_snow = crop_features(player(RAMS, GREEN)), crop_features(player(RAMS, SNOW))
    bears = crop_features(player(BEARS, SNOW))
    assert np.linalg.norm(on_grass - on_snow) < 0.5 * np.linalg.norm(on_snow - bears)


def test_teams_split_on_grass_and_snow_and_white_shirts_are_team_0():
    crops, truth = game()
    pred = ColourTeams().fit(crops).predict(crops)
    assert (pred == truth).all()


def test_ids_do_not_depend_on_crop_order():
    crops, truth = game(seed=1)
    order = np.random.default_rng(2).permutation(len(crops))
    pred = ColourTeams().fit([crops[i] for i in order]).predict(crops)
    assert (pred == truth).all()


def test_unusable_crops_predict_minus_one_and_margins_agree():
    crops, truth = game(seed=3)
    teams = ColourTeams().fit(crops)
    tiny = player(BEARS, GREEN, h=10, w=5)
    pred, margin = teams.predict([crops[0], tiny, crops[1]]), teams.margin([crops[0], tiny, crops[1]])
    assert list(pred) == [truth[0], -1, truth[1]]
    assert np.isnan(margin[1]) and (margin[[0, 2]] > 0).tolist() == [truth[0] == 1, truth[1] == 1]


def test_one_colour_cannot_be_split():
    with pytest.raises(ValueError):
        ColourTeams().fit([player(RAMS, GREEN, noise=0.0)] * 10)
    with pytest.raises(ValueError):
        ColourTeams().fit([player(RAMS, GREEN, h=10, w=5)] * 10)
