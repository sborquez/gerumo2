"""
Data and dataset constants
==========================
"""
from astropy.units.cds import rad, eV
import numpy as np

# Telescopes types
TELESCOPES = []

# Telescope array information
TELESCOPE_FEATURES = []
TELESCOPE_CAMERA   = {
}

# Regression Targets
REGRESSION_TARGETS = ["alt", "az", "mc_energy", "log10_mc_energy"]
REGRESSION_TARGET_UNITS = [rad, rad, 1e12*eV, np.log10(1e12)*eV]