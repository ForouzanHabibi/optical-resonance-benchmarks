"""Channel normalization and its analytic effect on 2D cylindrical scattering poles.

Lossless infinite dielectric cylinder, E_z polarization, exp(-i omega t).
The partial-wave incoming/outgoing scattering matrix is S_m(x) = 1+2*a_m(x),
where a_m is the exact coefficient in cylinder_mie.py, x=k0*R.

For an incoming channel rescaled by gamma_in and outgoing channel by gamma_out,
S_rescaled = gamma_in/gamma_out * S. We compare three normalizations:
  base: S0 = 1+2*a_m
  exponential: S1 = exp(2*i*x) * S0
  Hankel: S2 = H_m^(1)(x)/H_m^(2)(x) * S0.

The second normalization has channel poles at zeros of H_m^(2) independently
of the dielectric cylinder's outgoing resonances (zeros of D_m). The
cylinder geometry adapts the normalization mechanism discussed in
Foesleitner et al., Opt. Lett. 51, 2656-2659 (2026); it is not that paper's
spherical-scatterer calculation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import root
from scipy.special import hankel1, hankel2, h2vp
from cylinder_mie import cylinder_coefficient, cylinder_denominator, find_cylinder_pole
from cylinder_pole_expansion import cylinder_residue


def mie_s(m: int, x, n: float = 3.0):
    return 1.0 + 2.0 * cylinder_coefficient(m, x, n)


def channel_multiplier(m: int, x, basis: str):
    """Analytic ratio gamma_in/gamma_out of incoming and outgoing scales."""
    z = np.asarray(x, dtype=np.complex128)
    if basis == 'mie':
        return np.ones_like(z, dtype=np.complex128)
    if basis == 'exponential':
        return np.exp(2j*z)
    if basis == 'hankel':
        return hankel1(m, z) / hankel2(m, z)
    raise ValueError(f'Unknown channel normalization: {basis}')


def channel_scattering(m: int, x, n: float = 3.0, basis: str = 'mie'):
    return channel_multiplier(m, x, basis) * mie_s(m, x, n)


def field_channel_factors(m: int, x, basis: str):
    """Return (gamma_in, gamma_out) multiplying the Hankel basis functions.

    Incoming: H_m^(2); outgoing: H_m^(1). They multiply the basis functions,
    not the physical amplitudes. These factors recover the S transformations.
    """
    z = np.asarray(x, dtype=np.complex128)
    if basis == 'mie':
        return np.ones_like(z), np.ones_like(z)
    if basis == 'exponential':
        return np.exp(1j*z), np.exp(-1j*z)
    if basis == 'hankel':
        return 1/hankel2(m,z), 1/hankel1(m,z)
    raise ValueError(f'Unknown channel normalization: {basis}')


def find_channel_pole(m: int, guess: complex = 1.3 + 1.68j):
    """Locate a simple zero of the incoming cylindrical Hankel function H2."""
    def equations(v):
        val = hankel2(m, v[0]+1j*v[1]); return [float(val.real), float(val.imag)]
    solution = root(equations, [guess.real,guess.imag], tol=1e-12)
    p = complex(*solution.x)
    if (not solution.success or p.real <= 0 or p.imag <= 0
            or abs(hankel2(m,p))>1e-9 or abs(h2vp(m,p)) < 1e-10):
        raise RuntimeError(f'Channel zero not accurately found from guess {guess!r}')
    return p


def residue_physical_pole(m: int, pole: complex, n: float, basis: str):
    """Residue of rescaled S at outgoing resonance of cylinder D_m."""
    return complex(2*cylinder_residue(m,pole,n)*channel_multiplier(m,pole,basis))


def residue_channel_pole(m: int, pole: complex, n: float):
    """Residue of Hankel-rescaled S at a simple zero of incoming H2."""
    return complex(hankel1(m,pole)*mie_s(m,pole,n)/h2vp(m,pole))


def winding_count(values):
    """Argument-principle winding of analytic function on closed contour."""
    vals=np.asarray(values,dtype=np.complex128)
    if vals.size<8 or np.min(np.abs(vals))<1e-10:
        raise ValueError('Contour values too small for a reliable winding number')
    args=np.unwrap(np.angle(np.r_[vals,vals[0]]))
    count=(args[-1]-args[0])/(2*np.pi)
    if abs(count-round(count))>0.03:
        raise ValueError(f'Winding number ambiguous: {count}')
    return int(round(count))


@dataclass(frozen=True)
class ChannelExpansion:
    m: int
    n: float
    center: complex
    radius: float
    physical_poles: np.ndarray
    channel_poles: np.ndarray
    poles: np.ndarray
    residues: np.ndarray
    coefficients: np.ndarray
    physical_winding: int
    channel_winding: int

    def evaluate(self,x,background_order:int=20,include_channel:bool=True):
        z=np.asarray(x,dtype=np.complex128)
        if np.any(np.abs(z-self.center)>=self.radius):
            raise ValueError('Requested point is outside open expansion contour')
        if background_order<-1 or background_order>=len(self.coefficients):
            raise ValueError('Background order is not supported')
        result=np.zeros_like(z,dtype=np.complex128)
        count=len(self.physical_poles)+ (len(self.channel_poles) if include_channel else 0)
        for p,r in zip(self.poles[:count],self.residues[:count]):
            result+=r/(z-p)
        if background_order>=0:
            result+=np.polynomial.polynomial.polyval(z-self.center,
                                                     self.coefficients[:background_order+1])
        return result


def build_channel_expansion(m:int=3, n:float=3.0,
                            physical_seeds:tuple[complex,...]=(1.65-.018j,2.73-.059j,3.81-.085j),
                            channel_seeds:tuple[complex,...]=(1.30+1.68j,),
                            center:complex=2.5+0.5j, radius:float=1.9,
                            quadrature_points:int=4096,
                            max_background_order:int=30)->ChannelExpansion:
    """Construct local Cauchy expansion of the Hankel-normalized cylinder S.

    Independent contour-winding checks certify full D_m and H2 zero counts.
    An analytic background is computed from exact scattering on the contour.
    """
    center=complex(center)
    if center.real<=radius or radius<=0:
        raise ValueError('Contour must stay in Re(x)>0 away from Hankel branch cut')
    p=np.array([find_cylinder_pole(m,s,n) for s in physical_seeds],dtype=complex)
    q=np.array([find_channel_pole(m,s) for s in channel_seeds],dtype=complex)
    if any(abs(z-center)>=radius for z in np.r_[p,q]):
        raise ValueError('Found pole outside contour')
    if any(abs(u-v)<1e-5 for idx,u in enumerate(np.r_[p,q])
           for v in np.r_[p,q][:idx]):
        raise ValueError('Duplicate seeded pole')
    theta=2*np.pi*np.arange(quadrature_points)/quadrature_points
    contour=center+radius*np.exp(1j*theta)
    phys_count=winding_count(cylinder_denominator(m,contour,n))
    chan_count=winding_count(hankel2(m,contour))
    if (phys_count!=len(p) or chan_count!=len(q)):
        raise ValueError(f'Incomplete root list: physical {len(p)}/{phys_count}, '
                         f'channel {len(q)}/{chan_count}')
    p=p[np.argsort(p.real)]; q=q[np.argsort(q.real)]
    residues=np.r_[[residue_physical_pole(m,z,n,'hankel') for z in p],
                   [residue_channel_pole(m,z,n) for z in q]]
    exact_on_contour=channel_scattering(m,contour,n,'hankel')
    coeff=np.array([np.mean(exact_on_contour*np.exp(-1j*k*theta))/radius**k
                    for k in range(max_background_order+1)],dtype=complex)
    return ChannelExpansion(m,n,center,radius,p,q,np.r_[p,q],residues,coeff,
                            phys_count,chan_count)
