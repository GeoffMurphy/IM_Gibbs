"""Bayesian component separation and power spectrum estimation for 21cm
intensity mapping data cubes.

Gibbs sampling with Gaussian Constrained Realisations, jointly inferring the
21cm signal, the foreground amplitudes, and both of their covariances, on the
real (non-cubic) MeerKLASS L2021 footprint.
"""

from .data import DATA_DIR, data_path, load, load_l2021_cube
from .covariance import (
    foreground_covariance_sampler,
    signal_covariance_sampler,
)
from .grid import (
    DEFAULT_COSMO, FOOTPRINT_CROP, PIX_DEG, SurveyGrid, survey_grid,
)
from .kbins import (
    bin_it,
    footprint_kmin,
    kbins_from_crop,
    make_kbins,
    power_spectrum,
)
from .linear_system import (
    Uf,
    Us,
    construct_A,
    construct_Uf,
    construct_b,
    construct_preconditioner,
)

__version__ = '2.0.0.dev0'

__all__ = [
    'DATA_DIR', 'data_path', 'load', 'load_l2021_cube',
    'DEFAULT_COSMO', 'FOOTPRINT_CROP', 'PIX_DEG', 'SurveyGrid', 'survey_grid',
    'make_kbins', 'kbins_from_crop', 'footprint_kmin', 'bin_it',
    'power_spectrum',
    'Us', 'Uf', 'construct_Uf', 'construct_A', 'construct_b',
    'construct_preconditioner',
    'signal_covariance_sampler', 'foreground_covariance_sampler',
]
