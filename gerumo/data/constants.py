"""
Data and dataset constants
==========================
"""
from astropy.units.cds import rad, eV
import numpy as np
from ..utils.structures import Telescope

# Telescopes types
TELESCOPES = {
    "LST":  Telescope("LST", "LST", "LSTCam"),
    # "MST": Telescope("MST", "MST", "NectarCam"),
    "MST": Telescope("MST", "MST", "FlashCam"),
    "SST": Telescope("ASTRI", "SST", "CHEC")
}

# Telescope array information
HILLAS_PARAMETERS = [
    'hillas_intensity',
    'hillas_x',
    'hillas_y',
    'hillas_r',
    'hillas_phi',
    'hillas_length',
    'hillas_length_uncertainty',
    'hillas_width',
    'hillas_width_uncertainty',
    'hillas_psi',
    'hillas_skewness',
    'hillas_kurtosis'
]
TELESCOPE_FEATURES = [
    'pos_x',                # x array coordinate
    'pos_y',                # y array coordinate
    'pos_z',                # z array coordinate
] + HILLAS_PARAMETERS

# Regression Targets
REGRESSION_TARGETS = ["alt", "az", "mc_energy", "log10_mc_energy"]
REGRESSION_TARGET_UNITS = [rad, rad, 1e12*eV, np.log10(1e12)*eV]

# Classification Target
CLASSIFICATION_TARGET = ["particle_type"]
