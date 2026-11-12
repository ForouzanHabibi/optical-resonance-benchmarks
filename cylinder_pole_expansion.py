"""Local 2D cylindrical scattering pole-residue / analytic-background expansion.

The exact Mie coefficient a_m(x) is meromorphic ONLY locally away from the Hankel
branch point at x=0 and its branch cut. We apply Cauchy's theorem inside a disk
contained in Re(x)>0 and verify that every pole in that disk is included.

For |x-c|<rho,
 a_m(x) = sum_{p in disk} Res(a_m,p)/(x-p) + B_m(x),
 B_m(x) = sum_{k>=0} b_k (x-c)^k,
 b_k = (1/2pi i) integral_contour a_m(zeta)/(zeta-c)^(k+1) dzeta.

The background coefficients depend on values of exact Mie scattering on the
complex integration contour. This is a benchmark, NOT a predictive RSE solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
from cylinder_mie import cylinder_coefficient, cylinder_denominator, find_cylinder_pole


@dataclass(frozen=True)
class LocalPoleExpansion:
    m: int
    n: float
    center: complex
    radius: float
    poles: np.ndarray
    residues: np.ndarray
    bg_coeffs: np.ndarray
    pole_count_winding: int
    quadrature_points: int

    def evaluate(self, x, num_poles: int | None = None, background_order: int | None = None):
        """Evaluate truncated local expansion, pole set sorted nearest to disk center.

        If background_order is -1, include zero analytic background (pole-only).
        """
        x = np.asarray(x, dtype=np.complex128)
        if np.any(np.abs(x - self.center) >= self.radius):
            raise ValueError('Evaluation point outside the open Cauchy contour')
        if num_poles is None:
            num_poles = len(self.poles)
        if background_order is None:
            background_order = len(self.bg_coeffs)-1
        if not (0 <= num_poles <= len(self.poles)):
            raise ValueError('num_poles out of range')
        if not (-1 <= background_order < len(self.bg_coeffs)):
            raise ValueError('background_order out of range')
        result = np.zeros_like(x, dtype=np.complex128)
        for pole, residue in zip(self.poles[:num_poles], self.residues[:num_poles]):
            result += residue/(x-pole)
        if background_order >= 0:
            zz = x-self.center
            # Horner evaluation of sum b_k * (x-c)**k
            polynomial = np.zeros_like(x, dtype=np.complex128)
            for coeff in self.bg_coeffs[:background_order+1][::-1]:
                polynomial = polynomial*zz + coeff
            result += polynomial
        return result


def cylinder_residue(m: int, pole: complex, n: float = 3.0) -> complex:
    """Res[a_m, pole] = numerator(pole)/D_m'(pole).

    Differentiation cancels two cross terms in D_m'(x), giving
    D_m' = J_m(n*x) H_m''(x) - n**2 J_m''(n*x) H_m(x).
    """
    z = complex(pole)
    num = n*jv(m,z)*jvp(m,n*z) - jvp(m,z)*jv(m,n*z)
    dprime = jv(m,n*z)*h1vp(m,z,2)-n**2*jvp(m,n*z,2)*hankel1(m,z)
    if abs(dprime) < 1e-12:
        raise RuntimeError('Cannot determine reliable simple-pole residue')
    return complex(num/dprime)


def count_enclosed_denominator_zeros(m: int, n: float, center: complex,
                                     radius: float, quadrature_points: int=4096) -> int:
    """Argument-principle winding count, for disks avoiding Hankel branch cuts."""
    if center.real <= radius or abs(center.imag) > 1e-14:
        # intentionally restrict the demo to positive-real-centered disks
        raise ValueError('Contour must lie strictly in the right half-plane')
    theta=np.linspace(0, 2*np.pi, quadrature_points+1)
    z=center+radius*np.exp(1j*theta)
    D=cylinder_denominator(m,z,n)
    if np.min(np.abs(D)) < 1e-9:
        raise RuntimeError('Contour dangerously near an outgoing pole')
    phase_change = np.unwrap(np.angle(D))[-1] - np.unwrap(np.angle(D))[0]
    count=phase_change/(2*np.pi)
    if abs(count-round(count)) > .01:
        raise RuntimeError(f'Winding number is numerically ambiguous: {count}')
    return int(round(count))


def build_local_expansion(m: int, seeds: list[complex], n: float = 3.0,
                          center: float = 2.6, radius: float = 1.65,
                          max_background_order: int = 24,
                          quadrature_points: int = 4096) -> LocalPoleExpansion:
    """Build a *verified local* expansion from manually seeded physical poles.

    Does not claim global root discovery. In a right-half-plane disk, D_m is
    analytic and the argument principle certifies the number of zeros inside.
    """
    c=complex(center)
    if c.real <= radius or abs(c.imag) > 1e-14:
        raise ValueError('Circle must exclude Hankel branch point at x=0/cut')
    poles=np.asarray([find_cylinder_pole(m,s,n) for s in seeds],dtype=complex)
    if any(abs(p-c)>=radius for p in poles):
        raise ValueError('Some seeds converged to poles outside the contour')
    if any(abs(poles[i]-poles[j])<1e-6 for i in range(len(poles)) for j in range(i)):
        raise ValueError('Duplicate pole among the seeds')
    winding=count_enclosed_denominator_zeros(m,n,c,radius,quadrature_points)
    if winding != len(poles):
        raise ValueError(f'Found {len(poles)} seeded poles, but winding count indicates {winding}; ' 
                         'root set is incomplete or contour invalid')
    # Sort by distance to contour center for controlled term-count comparison.
    poles=poles[np.argsort(abs(poles-c))]
    residues=np.array([cylinder_residue(m,p,n) for p in poles])
    theta=2*np.pi*np.arange(quadrature_points)/quadrature_points
    contour=c+radius*np.exp(1j*theta)
    values=cylinder_coefficient(m,contour,n)
    # Uniform periodic trapezoid rule evaluates the Taylor coefficients of the
    # analytic contour background via a discrete Fourier transform.
    coeffs=np.array([np.mean(values*np.exp(-1j*k*theta))/radius**k
                      for k in range(max_background_order+1)],dtype=complex)
    return LocalPoleExpansion(m,n,c,radius,poles,residues,coeffs,winding,quadrature_points)


def complex_rmse(a,b) -> float:
    """Root mean squared complex amplitude deviation."""
    return float(np.sqrt(np.mean(np.abs(np.asarray(a)-np.asarray(b))**2)))
