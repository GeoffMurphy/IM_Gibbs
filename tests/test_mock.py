"""The mock observation and the chain that runs on it.

A mock is the answer every result is scored against, so the things that
must not drift are: the seeds (same seed, same cube; changing one component
leaves the others alone), the flagging convention (``cube != 0`` is the
mask), and the foreground staying inside the span the sampler models.

Run with:  pytest -q
"""

import numpy as np
import pytest

pytest.importorskip('pyccl', reason='geometry needs pyccl')
pytest.importorskip('fastbox', reason='mocks need fastbox')

from imgibbs import (                                          # noqa: E402
    Chain, drift_footprint, kbins_from_crop, legendre_basis, load_mock,
    make_mock, run_chain, signal_prior,
)
from imgibbs.mock import binned_power, component_seeds         # noqa: E402

SMALL = (24, 18, 48)


@pytest.fixture(scope='module')
def mock():
    return make_mock(SMALL, seed=5, systematic='groundspill')


def test_same_seed_same_mock(mock):
    again = make_mock(SMALL, seed=5, systematic='groundspill')
    assert np.array_equal(again.cube, mock.cube)


def test_components_have_independent_seeds(mock):
    """Switching the systematic off must not move the HI or the noise."""
    plain = make_mock(SMALL, seed=5, systematic=None)
    assert np.array_equal(plain.hi, mock.hi)
    assert np.array_equal(plain.foreground, mock.foreground)
    assert np.array_equal(plain.noise, mock.noise)
    assert not np.array_equal(plain.cube, mock.cube)
    assert len(set(component_seeds(5).values())) == 4


def test_building_a_mock_leaves_the_global_generator_alone():
    """FastBox seeds np.random; the chain's own stream must not move."""
    np.random.seed(11)
    expected = np.random.normal(size=3)
    np.random.seed(11)
    make_mock(SMALL, seed=1)
    assert np.array_equal(np.random.normal(size=3), expected)


def test_cube_is_the_sum_of_its_parts(mock):
    sky = mock.beam.apply(mock.hi + mock.foreground)
    expected = (sky + mock.systematic + mock.noise) * mock.mask
    obs = mock.mask > 0
    assert np.allclose(mock.cube[obs], expected[obs], rtol=0, atol=1e-12)
    assert np.all(mock.cube[~obs] == 0)


def test_zero_means_flagged(mock):
    assert np.array_equal(mock.cube != 0, mock.mask > 0)


def test_foreground_lies_inside_the_legendre_span(mock):
    """Power laws over a 5% band: 6 modes take them to rounding."""
    ev = legendre_basis(SMALL[2], 6)
    fg = mock.foreground.reshape(-1, SMALL[2])
    resid = fg - (fg @ ev.T) @ ev
    assert np.abs(resid).max() < 1e-8 * np.abs(fg).max()


def test_hi_is_in_kelvin(mock):
    """FastBox returns mK; S is in K^2. A 1e3 slip is a 1e6 slip in S."""
    assert 1e-4 < mock.hi.std() < 1e-2


def test_footprint_fill_is_drift_scan_like():
    fill = drift_footprint()[:, :, 0].mean()
    assert 0.5 < fill < 0.75


def test_save_and_load_round_trip(mock, tmp_path):
    path = tmp_path / 'mock.npz'
    mock.save(path)
    back = load_mock(path)
    assert np.array_equal(back.cube, mock.cube)
    assert np.array_equal(back.mask, mock.mask)
    assert np.array_equal(back.g_true, mock.g_true)
    assert np.allclose(back.beam.transfer, mock.beam.transfer)
    assert back.sys_basis.n_params == mock.sys_basis.n_params


def test_signal_prior_layout():
    idxs = np.array([0, 1, 1, 2, 0, 2])
    S = signal_prior([2.0, 3.0], idxs, dc_index=0)
    assert S[0] == 1e30                       # DC left free
    assert S[4] == 1e-12 * 2.5                # rest of bin 0 suppressed
    assert np.array_equal(S[[1, 2, 3, 5]], [2.0, 2.0, 3.0, 3.0])


def _chain(mock, seed):
    shape = mock.shape
    sig_k, idxs, meta = kbins_from_crop(mock.cube, mock.grid.box_dims,
                                        max_bins=3, min_modes=10,
                                        verbose=False)
    S = signal_prior(binned_power(mock.hi, sig_k, idxs), idxs,
                     meta['dc_index'])
    return run_chain(mock.cube, (mock.cube != 0).astype(float).ravel(),
                     legendre_basis(shape[2], 4), S, sig_k, idxs,
                     meta['dc_index'], mock.config['noise_rms'] ** 2, 2,
                     seed=seed, progress=False, log=lambda *a: None)


def test_chain_runs_and_is_reproducible(mock):
    a = _chain(mock, seed=3)
    b = _chain(mock, seed=3)
    assert isinstance(a, Chain)
    assert a.pk.shape[0] == 2 and np.all(np.isfinite(a.pk))
    assert np.array_equal(a.pk, b.pk)
    assert np.array_equal(a.F, b.F)
    assert a.signal_mean.shape == mock.shape
