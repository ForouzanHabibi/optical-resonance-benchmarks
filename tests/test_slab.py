"""Standard-library tests; run python -m unittest discover -s tests -v."""
import unittest
import numpy as np
from slab_resonances import (scattering, resonance_poles, denominator,
    truncated_pole_scattering, quality_factor, first_order_pole_shift)

class SlabBenchmarkTests(unittest.TestCase):
    def test_energy_conservation(self):
        x = np.linspace(0, 8, 1000)
        t, r = scattering(x)
        np.testing.assert_allclose(np.abs(t)**2 + np.abs(r)**2, 1.0, atol=2e-14)

    def test_complex_resonant_poles(self):
        p = resonance_poles(np.arange(-6, 7))
        self.assertLess(float(np.max(np.abs(denominator(p)))), 1e-12)
        self.assertTrue(np.all(p.imag < 0))

    def test_zero_frequency_anchor(self):
        t, r = truncated_pole_scattering(0.0, 10)
        self.assertAlmostEqual(abs(t-1.0), 0.0, places=12)
        self.assertAlmostEqual(abs(r), 0.0, places=12)

    def test_convergent_approximation(self):
        x = np.linspace(0.25, 2.5, 150)
        t, r = scattering(x)
        t8, r8 = truncated_pole_scattering(x, 8)
        t128, r128 = truncated_pole_scattering(x, 128)
        self.assertLess(float(np.linalg.norm(t-t128)), float(np.linalg.norm(t-t8)))
        self.assertLess(float(np.linalg.norm(r-r128)), float(np.linalg.norm(r-r8)))

    def test_perturbation_derivative(self):
        mode, n, dn = 4, 3.0, 1e-5
        exact = (resonance_poles(mode, n+dn) - resonance_poles(mode, n)).item()
        linear = first_order_pole_shift(mode, dn, n)
        self.assertLess(abs(exact-linear), 1e-9)

    def test_quality_factor(self):
        self.assertGreater(quality_factor(3), 0)

if __name__ == '__main__':
    unittest.main()
