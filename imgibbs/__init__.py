"""Bayesian component separation and power spectrum estimation for 21cm
intensity mapping data cubes.

Gibbs sampling with Gaussian Constrained Realisations, jointly inferring the
21cm signal, the foreground amplitudes, and both of their covariances, on a
non-cubic MeerKAT L-band grid. :mod:`imgibbs.mock` builds the observations it
runs on; :func:`run_chain` is the sampler.
"""

from .data import DATA_DIR, data_path, load
from .covariance import (
    foreground_covariance_sampler,
    signal_covariance_sampler,
)
from .grid import (
    DEFAULT_COSMO, PIX_DEG, SurveyGrid, survey_grid,
)
from .kbins import (
    bin_it,
    footprint_kmin,
    kbins_from_crop,
    make_kbins,
    power_spectrum,
)
from .systematics import (
    C_M_MHZ, ENVELOPE_INDEX, RIPPLE_PERIOD_MHZ, RM_DEFAULT, T_GROUND,
    SystematicBasis, TemplateBasis, beam_project, best_fit_amplitudes,
    deflate_foreground, project_out_beamed,
    faraday_templates, groundspill_basis,
    groundspill_cube, lambda_squared, leakage_basis, onef_basis,
    onef_covariance, period_scan, poly2d_templates, realise, ripple_wavenumber,
    scan_templates, spectral_templates, spillover_envelope,
)
from .beam import (
    FWHM_TO_SIGMA, Beam, beam_fwhm, chromatic_leakage, gaussian_beam,
    load_beam_table, radial_transfer, tabulated_beam, beam_from_spec,
    grid_beam, read_beam_profiles,
    beam_run_args,
)
from .linear_system import (
    MaskedForegroundBlock, MarginalNoise, rfft_adjoint_weight,
    Uf,
    Us,
    construct_A,
    construct_Uf,
    construct_b,
    construct_preconditioner,
)
from .mock import (
    Mock, drift_footprint, expected_hi_power, binned_power, foreground_cube,
    hi_cube, load_mock, make_mock, mock_grid, systematic_cube,
)
from .sampler import (
    Chain, legendre_basis, load_chain, run_chain, signal_prior,
)

__version__ = '2.0.0.dev0'

__all__ = [
    'DATA_DIR', 'data_path', 'load',
    'DEFAULT_COSMO', 'PIX_DEG', 'SurveyGrid', 'survey_grid',
    'make_kbins', 'kbins_from_crop', 'footprint_kmin', 'bin_it',
    'power_spectrum',
    'Us', 'Uf', 'construct_Uf', 'construct_A', 'construct_b',
    'construct_preconditioner', 'MaskedForegroundBlock', 'MarginalNoise',
    'rfft_adjoint_weight',
    'signal_covariance_sampler', 'foreground_covariance_sampler',
    'SystematicBasis', 'TemplateBasis', 'beam_project',
    'project_out_beamed', 'groundspill_basis', 'groundspill_cube',
    'spectral_templates', 'scan_templates', 'spillover_envelope',
    'ripple_wavenumber', 'period_scan',
    'T_GROUND', 'RIPPLE_PERIOD_MHZ', 'ENVELOPE_INDEX',
    'leakage_basis', 'faraday_templates', 'lambda_squared', 'poly2d_templates',
    'onef_basis', 'onef_covariance', 'best_fit_amplitudes', 'realise',
    'RM_DEFAULT', 'C_M_MHZ', 'deflate_foreground',
    'Beam', 'gaussian_beam', 'beam_fwhm', 'chromatic_leakage', 'FWHM_TO_SIGMA',
    'tabulated_beam', 'load_beam_table', 'radial_transfer',
    'grid_beam', 'read_beam_profiles',
    'beam_from_spec', 'beam_run_args',
    'Mock', 'make_mock', 'load_mock', 'mock_grid', 'drift_footprint',
    'hi_cube', 'foreground_cube', 'systematic_cube', 'binned_power',
    'expected_hi_power',
    'Chain', 'run_chain', 'load_chain', 'legendre_basis', 'signal_prior',
]
