"""Regression tests for the pieces that are easy to break silently.

The geometry and the k-binning are the two things a mistake in will not raise
an exception -- it will just attach the signal prior to the wrong wavenumbers
and produce a plausible-looking wrong answer. These pin them to the values the
shipped ``S_starting_point_cropped.npy`` was actually generated with.

Run with:  pytest -q
"""

import json

import numpy as np
import pytest

from imgibbs import (
    Us, bin_it, data_path, kbins_from_crop, make_kbins, power_spectrum,
    signal_covariance_sampler, survey_grid,
)
from imgibbs.grid import DEFAULT_COSMO

CROP = (slice(33, 103), slice(14, 59), slice(0, 250))
SHAPE = (70, 45, 250)

pytest.importorskip('pyccl', reason='geometry needs pyccl')


@pytest.fixture(scope='module')
def recorded():
    """Metadata written by the run that produced the shipped S."""
    path = data_path('S_starting_point_cropped_meta.json')
    if not path.exists():
        pytest.skip(f'{path} not present')
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_grid_reproduces_recorded_box_dims(recorded):
    grid = survey_grid(CROP, SHAPE)
    assert grid.shape == tuple(recorded['shape'])
    assert np.allclose(grid.box_dims, recorded['box_dims'], rtol=0, atol=0)
    assert grid.z_mid == recorded['z_mid']
    assert np.isclose(grid.freqs[0], recorded['freq_MHz'][0])
    assert np.isclose(grid.freqs[-1], recorded['freq_MHz'][1])


def test_grid_follows_the_crop():
    """A different frequency crop must move the radial extent with it."""
    full = survey_grid((slice(33, 103), slice(14, 59), slice(None)),
                       (70, 45, 500))
    half = survey_grid(CROP, SHAPE)
    assert full.shape[2] == 500 and half.shape[2] == 250
    assert full.box_dims[2] > half.box_dims[2]
    # Transverse extent depends on z_mid, which moves too, so it is not equal.
    assert full.box_dims[0] != half.box_dims[0]


def test_grid_rejects_inconsistent_shape():
    with pytest.raises(ValueError, match='frequency axis disagrees'):
        survey_grid(CROP, (70, 45, 123))


def test_voxels_are_strongly_anisotropic():
    vx, vy, vz = survey_grid(CROP, SHAPE).voxel
    assert vx / vz > 5, 'transverse/radial anisotropy is the whole reason ' \
                        'a cubic box_dims was wrong'


def test_default_cosmo_matches_fastbox():
    fastbox_box = pytest.importorskip('fastbox.box')
    assert dict(fastbox_box.default_cosmo) == DEFAULT_COSMO


# ---------------------------------------------------------------------------
# k-binning
# ---------------------------------------------------------------------------

def _band_cube(shape=SHAPE, width=0.35):
    """A cube with a diagonal band of non-zero voxels, like the drift scan."""
    nx, ny, nz = shape
    ii, jj = np.indices((nx, ny))
    band = np.abs(jj / ny - ii / nx) < width / 2
    cube = np.zeros(shape)
    cube[band] = 1.0
    return cube


def test_kbins_reproduce_the_recorded_run(recorded):
    """Requires the real cube; the footprint is measured from it."""
    path = data_path('L2021_polished_cube.npy')
    if not path.exists():
        pytest.skip('L2021_polished_cube.npy not available (not redistributed)')

    cube = np.load(path)[CROP]
    grid = survey_grid(CROP, cube.shape)
    sig_k, idxs, meta = kbins_from_crop(cube, grid.box_dims, max_bins=5,
                                        verbose=False)

    assert meta['n_k_bins'] == recorded['n_k_bins']
    assert meta['modes_per_bin'] == recorded['kbin_modes_per_bin']
    assert meta['k_min'] == recorded['kbin_k_min']
    assert meta['k_max'] == recorded['kbin_k_max']
    assert np.array_equal(sig_k, recorded['kbin_sig_k'])


def test_kbins_every_bin_clears_min_modes():
    cube = _band_cube()
    grid = survey_grid(CROP, cube.shape)
    _, _, meta = kbins_from_crop(cube, grid.box_dims, min_modes=20,
                                 max_bins=12, verbose=False)
    assert min(meta['modes_per_bin']) >= 20
    assert meta['n_k_bins'] <= 12


def test_kbins_max_bins_is_a_ceiling_not_a_count():
    cube = _band_cube()
    grid = survey_grid(CROP, cube.shape)
    _, _, lo = kbins_from_crop(cube, grid.box_dims, max_bins=5, verbose=False)
    _, _, hi = kbins_from_crop(cube, grid.box_dims, max_bins=200,
                               min_modes=20, verbose=False)
    assert lo['n_k_bins'] <= 5
    assert hi['n_k_bins'] <= 200


def test_kbins_drops_the_kz0_plane():
    cube = _band_cube()
    grid = survey_grid(CROP, cube.shape)
    _, idxs, _ = kbins_from_crop(cube, grid.box_dims, drop_kz0=True,
                                 verbose=False)
    nz = cube.shape[2]
    kz = np.abs(2 * np.pi * np.fft.fftfreq(nz) * nz / grid.box_dims[2])
    KZ = np.broadcast_to(kz, cube.shape).ravel()
    assert np.all(idxs[KZ == 0] == 0), 'kz=0 modes must be excluded'


def test_kbins_lowest_edge_is_pinned():
    """logspace does not round-trip; an unpinned edge silently drops modes."""
    cube = _band_cube()
    grid = survey_grid(CROP, cube.shape)
    _, idxs, meta = kbins_from_crop(cube, grid.box_dims, verbose=False)
    assert (idxs > 0).sum() + meta['n_excluded'] == idxs.size


def test_dc_index_is_the_dc_mode():
    cube = _band_cube()
    grid = survey_grid(CROP, cube.shape)
    _, idxs, meta = kbins_from_crop(cube, grid.box_dims, verbose=False)
    assert meta['dc_index'] == 0          # fftfreq puts DC first
    assert idxs[meta['dc_index']] == 0


def test_make_kbins_bins_are_contiguous_and_cover_everything():
    sig_k, idxs = make_kbins((16, 12, 20), 6, box_dims=(100.0, 80.0, 50.0))
    assert len(sig_k) == 6
    assert idxs.min() == 0 and idxs.max() == 6
    assert idxs.size == 16 * 12 * 20


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def test_bin_it_uses_the_right_k_for_each_bin():
    """sig_k must be indexed by bin id, not by position, or an empty middle
    bin shifts every k value after it."""
    idxs = np.array([0, 1, 1, 1, 3, 3, 3])        # bin 2 is empty
    sig_k = np.array([0.1, 0.2, 0.3])
    cube = np.arange(7, dtype=float)
    binned, kb = bin_it(cube, sig_k, idxs)
    assert len(binned) == 2
    assert np.allclose(kb[0], 0.1)
    assert np.allclose(kb[1], 0.3), 'bin 3 must carry sig_k[2], not sig_k[1]'


def test_power_spectrum_recovers_a_white_field():
    """White noise of variance s^2 has P(k) = s^2 * V_voxel, flat in k."""
    rng = np.random.default_rng(0)
    shape, box = (24, 24, 24), (100.0, 100.0, 100.0)
    sigma = 0.3
    cube = rng.normal(0.0, sigma, shape)
    sig_k, idxs = make_kbins(shape, 5, box_dims=box)
    Pk, _, n = power_spectrum(cube, sig_k, idxs, box)
    expected = sigma**2 * np.prod(box) / np.prod(shape)
    assert np.allclose(Pk[n > 50], expected, rtol=0.25)


def test_power_spectrum_mask_uses_valid_voxels_only():
    """Subtracting a whole-cube mean would stamp the footprint into the field
    and dump spurious power at low k."""
    rng = np.random.default_rng(1)
    shape, box = (24, 24, 24), (100.0, 100.0, 100.0)
    cube = rng.normal(5.0, 0.3, shape)            # large offset
    mask = np.zeros(shape, dtype=bool)
    mask[:12] = True
    cube[~mask] = 0.0
    sig_k, idxs = make_kbins(shape, 5, box_dims=box)
    Pk_masked, _, _ = power_spectrum(cube, sig_k, idxs, box, mask=mask)
    Pk_naive, _, _ = power_spectrum(cube, sig_k, idxs, box, mask=None)
    assert Pk_naive[0] > 10 * Pk_masked[0]


def test_power_spectrum_cross_of_a_field_with_itself_is_its_auto():
    rng = np.random.default_rng(2)
    shape, box = (16, 16, 16), (50.0, 50.0, 50.0)
    cube = rng.normal(size=shape)
    sig_k, idxs = make_kbins(shape, 4, box_dims=box)
    auto, _, _ = power_spectrum(cube, sig_k, idxs, box)
    cross, _, _ = power_spectrum(cube, sig_k, idxs, box, cube2=cube)
    assert np.allclose(auto, cross)


# ---------------------------------------------------------------------------
# Operators and samplers
# ---------------------------------------------------------------------------

def test_Us_round_trips_on_an_even_channel_count():
    rng = np.random.default_rng(3)
    cube = rng.normal(size=(8, 6, 10))
    assert np.allclose(Us(Us(cube, True), False), cube)


def test_Us_loses_a_channel_on_an_odd_count():
    """Documented landmine: irfftn is called without s=, so it always returns
    an even last axis. Dormant at 250/500 channels, fatal if re-channelised."""
    rng = np.random.default_rng(4)
    cube = rng.normal(size=(8, 6, 11))
    assert Us(Us(cube, True), False).shape == (8, 6, 10)


def test_signal_covariance_sampler_is_in_the_right_ballpark():
    """The inverse-gamma draw should sit near the empirical mode variance."""
    rng = np.random.default_rng(5)
    n, true_var = 4000, 2.0
    s = rng.normal(0, np.sqrt(true_var), n)
    k = np.full(n, 0.1)
    Pk, per_bin = signal_covariance_sampler(s, k)
    assert len(per_bin) == 1
    assert 0.8 * true_var < per_bin[0] < 1.25 * true_var
    assert np.allclose(Pk, per_bin[0])


def test_signal_covariance_sampler_rejects_thin_bins():
    s = np.arange(2, dtype=float)
    k = np.zeros(2)
    with pytest.raises(AssertionError, match='greater than 2'):
        signal_covariance_sampler(s, k)
