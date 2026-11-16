"""Tests for the tau-space EnKF solver (solver/TAUENKF.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from compflowlab.solver import TAUENKF


def sod_like(x, x_rare_head=0.3, x_rare_tail=0.45, x_contact=0.6, x_shock=0.75):
    """Piecewise profile with the Sod structure: rarefaction, plateau, contact, plateau, shock."""
    f = np.where(x < x_rare_head, 1.0, 0.0)
    ramp = (x >= x_rare_head) & (x < x_rare_tail)
    f[ramp] = 1.0 - 0.6 * (x[ramp] - x_rare_head) / (x_rare_tail - x_rare_head)
    f[(x >= x_rare_tail) & (x < x_contact)] = 0.4
    f[(x >= x_contact) & (x < x_shock)] = 0.3
    f[x >= x_shock] = 0.125
    return f


def test_find_cuts_places_cuts_on_plateaus():
    x = np.linspace(0, 1, 1001)
    f = sod_like(x)
    cuts = TAUENKF.find_cuts(x, f, n_jumps=2)
    assert len(cuts) == 2
    assert 0.45 < cuts[0] < 0.6      # between rarefaction tail and contact
    assert 0.6 < cuts[1] < 0.75      # between contact and shock


def test_find_cuts_ignores_steep_rarefaction():
    """A weak shock less steep than the rarefaction must still get its own region."""
    x = np.linspace(0, 1, 1001)
    f = sod_like(x, x_rare_head=0.40, x_rare_tail=0.45)   # very steep rarefaction
    f[(x >= 0.6)] = 0.38                                   # weak contact/shock
    cuts = TAUENKF.find_cuts(x, f, n_jumps=1)
    assert 0.45 < cuts[0] < 0.6


def test_find_cuts_initial_step_falls_back_to_left_state():
    x = np.linspace(0, 1, 1001)
    f = np.where(x < 0.5, 1.0, 0.1)
    cuts = TAUENKF.find_cuts(x, f, n_jumps=1)
    assert cuts[0] == pytest.approx(0.25, abs=0.01)


def test_tau_state_roundtrip():
    x = np.linspace(0, 1, 501)
    f = sod_like(x)
    s_levels = np.linspace(0, 1, 400)
    z = TAUENKF.to_tau_state(x, f, 2, s_levels)
    assert z.shape == (2 * 3 * 400,)
    f_back = TAUENKF.from_tau_state(z, x)
    assert np.sum(np.abs(f_back - f)) * (x[1] - x[0]) < 5e-3


def test_enkf_pulls_ensemble_towards_observation():
    rng = np.random.default_rng(0)
    N, n = 200, 3
    Z = rng.normal(0.0, 1.0, (N, n))
    HZ = Z[:, :1]
    y = np.array([2.0])
    Za = TAUENKF.enkf(Z, HZ, y, np.array([0.1]), rng)
    assert abs(Za[:, 0].mean() - 2.0) < 0.1
    assert Za[:, 0].std() < 0.3
