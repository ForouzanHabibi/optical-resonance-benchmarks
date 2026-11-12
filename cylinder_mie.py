"""Independent 2D TM_z benchmark: an infinite homogeneous dielectric cylinder.

Time convention exp(-i omega t); outgoing waves use Hankel H_m^(1).
Ambient n=1; cylinder n>1; dimensionless x=k0*radius.
Requires scipy for Bessel/Hankel functions and complex root search.
"""
from __future__ import annotations
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.optimize import root


def cylinder_denominator(m: int, x, n: float = 3.0):
    """Outgoing-mode boundary condition for 2D E_z polarization."""
    x = np.asarray(x, dtype=np.complex128)
    return jv(m, n*x)*h1vp(m, x) - n*jvp(m, n*x)*hankel1(m, x)


def cylinder_coefficient(m: int, x, n: float = 3.0):
    """TM_z cylindrical scattering coefficient a_m for an incident plane wave."""
    x = np.asarray(x, dtype=np.complex128)
    numerator = n*jv(m, x)*jvp(m, n*x) - jvp(m, x)*jv(m, n*x)
    return numerator / cylinder_denominator(m, x, n)


def cylinder_scattering_efficiency(x, n: float = 3.0, max_m: int = 18):
    """Q_sca = C_sca/(2R) = (2/x) * (|a_0|^2+2*sum_m>0|a_m|^2)."""
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError('x must be positive for a scattering efficiency')
    weighted = np.abs(cylinder_coefficient(0, x, n))**2
    for m in range(1, max_m+1):
        weighted += 2*np.abs(cylinder_coefficient(m, x, n))**2
    return 2*weighted/x


def find_cylinder_pole(m: int, guess: complex, n: float = 3.0):
    """Seek one outgoing pole from a user-chosen initial complex-frequency guess.

    The nonlinear root finder is local: no guarantee of finding every pole.
    """
    def vec(y):
        val = cylinder_denominator(m, y[0]+1j*y[1], n)
        return [val.real, val.imag]
    sol = root(vec, [guess.real, guess.imag], tol=1e-11)
    pole = complex(sol.x[0], sol.x[1])
    if (not sol.success or abs(cylinder_denominator(m, pole, n)) > 1e-8
            or pole.real <= 0 or pole.imag >= 0):
        raise RuntimeError(f'No reliable outgoing pole for m={m}, guess={guess}')
    return pole
