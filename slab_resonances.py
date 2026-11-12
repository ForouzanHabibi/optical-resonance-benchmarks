"""Exactly solvable 1D optical slab: scattering amplitudes and resonant poles.

Convention: e^{-i omega t}; dimensionless spectral coordinate x = k_0 d.
A lossless n>1 slab (thickness d) is surrounded by air on both sides.
Amplitudes are referred to the two slab interfaces, not a global far-field origin.
"""
from __future__ import annotations

import numpy as np


def _check_index(n: float) -> float:
    n = float(n)
    if not np.isfinite(n) or n <= 1.0:
        raise ValueError("This analytic resonance branch requires real n > 1")
    return n


def scattering(x, n: float = 3.0):
    """Return complex transmission (t) and reflection (r) amplitudes.

    x = k0*d can be real (spectrum) or complex (pole verification).
    """
    n = _check_index(n)
    x = np.asarray(x, dtype=np.complex128)
    A = (n*n + 1.0)/(2.0*n)
    B = (n*n - 1.0)/(2.0*n)
    z = n*x
    denominator = np.cos(z) - 1j*A*np.sin(z)
    return 1.0/denominator, (1j*B*np.sin(z))/denominator


def denominator(x, n: float = 3.0):
    """Inverse transmission denominator: its complex zeros are scattering poles."""
    n = _check_index(n)
    A = (n*n + 1.0)/(2.0*n)
    return np.cos(n*x) - 1j*A*np.sin(n*x)


def resonance_poles(mode_numbers, n: float = 3.0):
    """Analytic outgoing poles x_m=(m*pi - i*alpha)/n, m integer."""
    n = _check_index(n)
    m = np.asarray(mode_numbers, dtype=np.int64)
    alpha = np.log((n+1.0)/(n-1.0))
    return (m*np.pi - 1j*alpha)/n


def pole_residues(mode_numbers, n: float = 3.0):
    """Exact residues (Res(t), Res(r)) at the slab's outgoing poles."""
    n = _check_index(n)
    m = np.asarray(mode_numbers, dtype=np.int64)
    B = (n*n - 1.0)/(2.0*n)
    residue_t = 1j * (-1.0)**m / (n*B)
    residue_r = 1j * np.ones_like(m, dtype=np.complex128) / (n*B)
    return residue_t, residue_r


def truncated_pole_scattering(x, modes_per_side: int, n: float = 3.0):
    """Anchored symmetric truncated Mittag-Leffler-style pole series.

    Uses exact residues and the exact x=0 reference t(0)=1, r(0)=0.
    Subtraction 1/(x-x_m) - 1/(0-x_m) stabilizes convergence.
    Note: A pole-series benchmark, NOT a full Maxwell resonant-state expansion.
    """
    n = _check_index(n)
    if not isinstance(modes_per_side, int) or modes_per_side < 0:
        raise ValueError("modes_per_side must be a nonnegative integer")
    inp = np.asarray(x, dtype=np.complex128)
    orig_shape = inp.shape
    values = np.atleast_1d(inp).reshape(-1)
    m = np.arange(-modes_per_side, modes_per_side + 1)
    poles = resonance_poles(m, n)
    residue_t, residue_r = pole_residues(m, n)
    kernels = 1.0/(values[:, None] - poles[None, :]) + 1.0/poles[None, :]
    approx_t = 1.0 + kernels @ residue_t
    approx_r = kernels @ residue_r
    return approx_t.reshape(orig_shape), approx_r.reshape(orig_shape)


def quality_factor(mode_number: int, n: float = 3.0):
    """Radiative Q for a pole with Re(x)>0."""
    pole = resonance_poles(np.asarray(mode_number), n).item()
    if pole.real <= 0.0:
        raise ValueError("Select a positive-frequency mode (m >= 1)")
    return -pole.real/(2.0*pole.imag)


def first_order_pole_shift(mode_number: int, delta_n: float, n: float = 3.0):
    """Derivative-based first-order shift of x_m for a small index perturbation."""
    n = _check_index(n)
    x_m = resonance_poles(np.asarray(mode_number), n).item()
    dxdn = -x_m/n + 2j/(n*(n*n-1.0))
    return dxdn*delta_n
