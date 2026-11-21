"""compute_fad.py のテスト（PANNs の重みは使わない）。 実行: .venv/bin/python -m pytest -q"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import soundfile as sf
import torch

import compute_fad as c


def reference_fad(x: np.ndarray, y: np.ndarray) -> float:
    """元の式そのまま（float64、sqrtm）"""
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    s, _ = scipy.linalg.sqrtm(cx @ cy, disp=False)
    return float((mx - my) @ (mx - my) + np.trace(cx) + np.trace(cy) - 2 * np.trace(s.real))


# MARK: - FAD の式

@pytest.mark.parametrize("n_x,n_y,d", [(500, 400, 64), (3000, 2500, 256)])
def test_matches_original_formula_when_full_rank(n_x, n_y, d):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_x, d))
    y = rng.standard_normal((n_y, d)) @ rng.standard_normal((d, d)) * 0.3 + 0.5
    assert c.frechet_distance(x, y)["fad"] == pytest.approx(reference_fad(x, y), rel=1e-9)


@pytest.mark.parametrize("n_x,n_y", [(4, 4), (24, 24), (50, 4)])
def test_matches_kadtk_float64_when_rank_deficient(n_x, n_y):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n_x, 2048)).astype(np.float32)
    y = (rng.standard_normal((n_y, 2048)) * 1.3 + 0.2).astype(np.float32)
    ours = c.frechet_distance(x, y)["fad"]
    assert ours == pytest.approx(c.kadtk_frechet_distance(x, y, precision=torch.float64), rel=1e-6)
    assert ours == pytest.approx(reference_fad(x.astype(np.float64), y.astype(np.float64)), rel=1e-6)


def test_basic_properties():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((24, 2048))
    y = rng.standard_normal((24, 2048)) * 0.7
    assert abs(c.frechet_distance(x, x)["fad"]) < 1e-8                       # 同じ集合は 0
    assert c.frechet_distance(x, y)["fad"] == pytest.approx(c.frechet_distance(y, x)["fad"])  # 対称
    shift = np.full(2048, 0.01)
    d0, d1 = c.frechet_distance(x, x)["fad"], c.frechet_distance(x, x + shift)["fad"]
    assert d1 - d0 == pytest.approx(shift @ shift)                           # 平行移動は平均の項だけ増える
    assert c.frechet_distance(x, y)["fad"] >= 0


# MARK: - 前処理

def test_load_clip_mixes_to_mono_and_resamples(tmp_path):
    sr = 44100
    t = np.arange(sr) / sr
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(tmp_path / "a.wav", np.stack([tone, tone], axis=1), sr, subtype="PCM_16")
    clip = c.load_clip(tmp_path / "a.wav", 22050)
    assert clip.x.ndim == 1 and clip.x.shape[0] == 22050
    assert clip.info["orig_sr"] == sr and clip.info["orig_channels"] == 2
    assert np.max(np.abs(clip.x)) == pytest.approx(0.5, abs=0.01)           # 振幅は変えない


def _clip(name, n, value=0.1):
    return c.Clip(name, np.full(n, value, dtype=np.float32), {"file": f"{name}.wav"})


def test_align_shortest_by_name():
    sets = {"ref": [_clip("1", 80000), _clip("2", 30000)],
            "nvidia": [_clip("1", 68096), _clip("2", 36096)],
            "iphone": [_clip("1", 67072), _clip("2", 36096)]}
    lengths = c.align_lengths(sets, "shortest-by-name", 22050, None)
    assert lengths == {"1": 67072, "2": 30000}
    assert all(cl.x.shape[0] == lengths[cl.name] for clips in sets.values() for cl in clips)


def test_align_rejects_mismatched_names():
    sets = {"ref": [_clip("1", 100), _clip("2", 100)], "iphone": [_clip("1", 100)]}
    with pytest.raises(SystemExit):
        c.align_lengths(sets, "shortest-by-name", 22050, None)


def test_align_fixed_never_pads():
    sets = {"ref": [_clip("1", 100)], "iphone": [_clip("1", 50)]}
    with pytest.raises(SystemExit):
        c.align_lengths(sets, "fixed", 100, 0.8)


# MARK: - 標本の独立性

def test_repeated_outputs_are_rejected():
    clips = [_clip("1_r1", 100, 0.1), _clip("1_r2", 100, 0.1), _clip("6_r1", 100, 0.2)]
    with pytest.raises(SystemExit):
        c.check_independent(clips, "iphone")
    c.check_independent([clips[0], clips[2]], "iphone")
