"""Numerical validations for exact 2D cylinder scattering and pole conditions."""
import unittest
import numpy as np
from cylinder_mie import (cylinder_coefficient, cylinder_denominator,
                          cylinder_scattering_efficiency, find_cylinder_pole)

class CylinderTests(unittest.TestCase):
    def test_channel_unitarity(self):
        x=np.linspace(.2, 6.0, 101)
        for m in range(6):
            s=1+2*cylinder_coefficient(m,x)
            np.testing.assert_allclose(np.abs(s), np.ones_like(x),atol=2e-12)

    def test_scattering_is_nonnegative(self):
        x=np.linspace(.2,5.0,35)
        self.assertTrue(np.all(cylinder_scattering_efficiency(x)>=0))

    def test_outgoing_pole_boundary_condition(self):
        pole=find_cylinder_pole(2,1.21-.04j)
        self.assertLess(abs(cylinder_denominator(2,pole)),1e-9)
        self.assertLess(pole.imag,0)

if __name__=='__main__':
    unittest.main()
