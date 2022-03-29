"""
Data and dataset constants
==========================
"""
import astropy.units as u
from ..utils.structures import Telescope

# Telescopes types
TELESCOPES = {
    'LST': Telescope('LST', 'LST', 'LSTCam'),
    # "MST": Telescope("MST", "MST", "NectarCam"),
    'MST': Telescope('MST', 'MST', 'FlashCam'),
    'SST': Telescope('ASTRI', 'SST', 'CHEC')
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

TELESCOPE_TIME_PEAK_MAX = {
    'LST': 3510.0,
    'MST': 7687.0,
    'SST': 11410.0
}

# Regression Targets
REGRESSION_TARGETS = ['az', 'alt', 'mc_energy', 'log10_mc_energy']
REGRESSION_TARGET_UNITS = [u.deg, u.deg, u.TeV, u.TeV]

# Classification Target
CLASSIFICATION_TARGET = ['particle_type']
