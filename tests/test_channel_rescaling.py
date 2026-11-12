"""Numerical checks for physical and channel-induced poles of cylindrical S."""
import unittest
import numpy as np
from scipy.special import hankel1,hankel2
from cylinder_mie import cylinder_denominator
from cylinder_pole_expansion import complex_rmse
from channel_rescaling import (
    mie_s, channel_multiplier, channel_scattering, field_channel_factors,
    find_channel_pole, build_channel_expansion, residue_channel_pole,
    residue_physical_pole,
)


class ChannelNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=3; cls.n=3.
        cls.e=build_channel_expansion(m=3,n=3.)
        cls.real=np.linspace(1.5,3.5,201)

    def test_all_physical_resonances_are_in_lower_half_plane(self):
        self.assertEqual(self.e.physical_winding,3)
        self.assertTrue(np.all(self.e.physical_poles.imag<0))
        for p in self.e.physical_poles:
            self.assertLess(abs(cylinder_denominator(self.m,p,self.n)),1.e-8)

    def test_channel_pole_is_in_upper_half_plane_and_not_resonance(self):
        q=self.e.channel_poles[0]
        self.assertGreater(q.imag,0)
        self.assertLess(abs(hankel2(self.m,q)),1e-9)
        self.assertGreater(abs(cylinder_denominator(self.m,q,self.n)),1e-4)
        self.assertEqual(self.e.channel_winding,1)

    def test_hankel_rescale_has_singular_channel_pole(self):
        q=self.e.channel_poles[0]
        z=q+1.e-6*(1+1j)
        self.assertGreater(abs(channel_scattering(self.m,z,self.n,'hankel')),1.e4)
        self.assertLess(abs(channel_scattering(self.m,z,self.n,'mie')),100.)
        self.assertLess(abs(channel_scattering(self.m,z,self.n,'exponential')),100.)

    def test_physical_s_is_unitary_on_real_axis(self):
        for basis in ('mie','exponential','hankel'):
            s=channel_scattering(self.m,self.real,self.n,basis)
            self.assertLess(float(np.max(np.abs(np.abs(s)-1))),1e-11)

    def test_channel_normalization_recovers_original_s(self):
        base=mie_s(self.m,self.real,self.n)
        for basis in ('mie','exponential','hankel'):
            scaled=channel_scattering(self.m,self.real,self.n,basis)
            back=scaled/channel_multiplier(self.m,self.real,basis)
            self.assertLess(float(np.max(abs(back-base))),2e-14)
            gin,gout=field_channel_factors(self.m,self.real,basis)
            self.assertLess(float(np.max(abs(channel_multiplier(self.m,self.real,basis)-gin/gout))),2e-14)

    def test_channel_basis_reconstructs_identical_physical_fields(self):
        x=2.3; r_over_R=1.43
        incoming=hankel2(self.m,x*r_over_R)
        outgoing=hankel1(self.m,x*r_over_R)
        s0=mie_s(self.m,x,self.n)
        original=incoming+s0*outgoing
        for basis in ('mie','exponential','hankel'):
            gin,gout=field_channel_factors(self.m,x,basis)
            S=channel_scattering(self.m,x,self.n,basis)
            # A'=A/gin and B'=B/gout; S'=B'/A'.
            Anew=1/gin; Bnew=S*Anew
            field=Anew*gin*incoming+Bnew*gout*outgoing
            self.assertLess(abs(field-original),2.e-13)

    def test_channel_residue_vs_numerical_limit(self):
        q=self.e.channel_poles[0]
        h=3e-7*(1+1j)
        direct=h*channel_scattering(self.m,q+h,self.n,'hankel')
        calculated=residue_channel_pole(self.m,q,self.n)
        self.assertLess(abs(direct-calculated)/abs(calculated),1e-5)

    def test_physical_residues_survive_both_normalizations(self):
        for p in self.e.physical_poles:
            for basis in ('mie','exponential','hankel'):
                h=2.e-7*(1+1j)
                estimated=h*channel_scattering(self.m,p+h,self.n,basis)
                analytical=residue_physical_pole(self.m,p,self.n,basis)
                self.assertLess(abs(estimated-analytical)/(abs(analytical)+1e-10),2e-5)

    def test_all_four_poles_reconstruct_complex_s(self):
        truth=channel_scattering(self.m,self.real,self.n,'hankel')
        predicted=self.e.evaluate(self.real,background_order=22,include_channel=True)
        self.assertLess(complex_rmse(predicted,truth),2e-8)

    def test_channel_pole_omission_prevents_convergence(self):
        truth=channel_scattering(self.m,self.real,self.n,'hankel')
        full=complex_rmse(self.e.evaluate(self.real,22,True),truth)
        omitted=complex_rmse(self.e.evaluate(self.real,22,False),truth)
        self.assertGreater(omitted,0.05)
        self.assertGreater(omitted/full,1.e6)

    def test_incomplete_channel_pole_roots_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete root list'):
            build_channel_expansion(channel_seeds=())

    def test_invalid_basis_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown channel'):
            channel_multiplier(3,2.4,'invalid')


if __name__=='__main__':unittest.main()
