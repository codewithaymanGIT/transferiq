"""Tests for the PSI helper in scripts/train/stability_psi.py (synthetic data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "train"))

from stability_psi import label, psi  # noqa: E402


def test_identical_distributions_have_near_zero_psi():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x) < 1e-9


def test_shifted_distribution_is_flagged():
    rng = np.random.default_rng(1)
    expected = rng.normal(0, 1, 5000)
    actual = rng.normal(1.5, 1, 5000)
    assert psi(expected, actual) > 0.25


def test_small_sample_noise_stays_stable():
    rng = np.random.default_rng(2)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.10


def test_handles_many_zeros_and_nans():
    expected = np.array([0, 0, 0, 0, 1, 2, 3, np.nan, 5, 8] * 20, dtype=float)
    actual = np.array([0, 0, 1, np.nan, 2, 2, 3, 4, 9, 30] * 20, dtype=float)
    value = psi(expected, actual)
    assert np.isfinite(value) and value >= 0


def test_live_values_outside_training_range_are_counted():
    expected = np.linspace(0, 1, 1000)
    actual = np.full(1000, 50.0)  # everything above the training max
    assert psi(expected, actual) > 0.25


def test_binary_flag_shift_is_detected():
    expected = np.array([1] * 30 + [0] * 70, dtype=float)   # 30% flagged
    actual = np.array([1] * 70 + [0] * 30, dtype=float)     # 70% flagged
    assert psi(expected, actual) > 0.25
    assert psi(expected, expected) < 1e-9


def test_labels_follow_rule_of_thumb():
    assert label(0.05) == "stable"
    assert label(0.18) == "moderate shift"
    assert label(0.4) == "significant shift"
    assert label(float("nan")) == "n/a"
