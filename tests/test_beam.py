"""Tests for the beam in the forward model.

Three groups, in order of how much rests on them.

The operator: ``B`` is a symmetric, mean-preserving, channel-by-channel
Gaussian convolution, checked against a directly sampled kernel rather than
against itself.

The linear system: ``beam=None`` is the published sampler untouched; an
identity beam reproduces it to rounding, which validates the restructured
beam path against the block-by-block one; and every row of ``Ax`` and ``b``
matches the normal equations written out by hand. The preconditioner's
foreground block is exact for uniform noise and its signal block is exact for
an achromatic beam.

The physics: a chromatic beam pushes a spectrally smooth sky outside the
foreground span, an achromatic one does not, and a spatially uniform sky is
untouched either way. That is the mode-mixing the beam was added to model.

Run with:  pytest -q tests/test_beam.py
"""

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, lgmres

from imgibbs import (
    FWHM_TO_SIGMA, Beam, beam_fwhm, chromatic_leakage, construct_A,
    construct_b, construct_preconditioner, gaussian_beam, groundspill_basis,
    beam_project, project_out_beamed, MarginalNoise, Us, Uf,
    rfft_adjoint_weight,
)

SHAPE = (8, 6, 16)
FREQS = 970.94 + np.arange(SHAPE[2]) * 0.208984 * 15
N_MODES = 3
PIX = 0.3

REAL_FREQS = 970.94140625 + np.arange(250) * (1712.0 - 856.0) / 4096


def legendre_basis(n_freq, n_modes):
    vander = np.polynomial.legendre.legvander(np.linspace(-1, 1, n_freq),
                                              n_modes - 1)
    return np.linalg.qr(vander)[0].T


def chromatic(shape=SHAPE, freqs=FREQS, index=1.0, fwhm=0.6):
    """A beam a couple of pixels wide, so it does something on 8 x 6."""
    return gaussian_beam(freqs, shape, fwhm_ref=fwhm, nu_ref=freqs[0],
                         index=index, pix_deg=PIX)


def identity_beam(shape=SHAPE):
    nx, ny, nz = shape
    return Beam(np.ones((nx, ny // 2 + 1, nz)), shape)


@pytest.fixture(scope='module')
def system():
    nx, ny, nz = SHAPE
    rng = np.random.default_rng(0)
    basis = groundspill_basis(FREQS, SHAPE, period=40.0)
    return dict(
        basis=basis,
        evecs=legendre_basis(nz, N_MODES),
        S=np.abs(rng.normal(size=nx * ny * nz)) + 1.0,
        F=np.abs(rng.normal(size=N_MODES)) + 1.0,
        G=np.abs(rng.normal(size=basis.n_params)) + 1.0,
        w=np.ones(nx * ny * nz),
        Nw_inv=np.full(nx * ny * nz, 2.0),
        rfft_len=nx * ny * (nz // 2 + 1),
        rfft_shape=(nx, ny, nz // 2 + 1),
        f_len=nx * ny * N_MODES,
        f_shape=(nx, ny, N_MODES),
        shape=SHAPE,
    )


def A_of(s, x, **kw):
    return construct_A(x, s['S'], s['Nw_inv'], s['F'], s['w'], s['evecs'],
                       s['rfft_len'], s['rfft_shape'], s['f_len'],
                       s['f_shape'], s['shape'], **kw)


def dense_A(s, **kw):
    n = 2 * s['rfft_len'] + s['f_len']
    if kw.get('sys_basis') is not None:
        n += kw['sys_basis'].n_params
    return np.array([A_of(s, np.eye(n)[i], **kw) for i in range(n)]).T


def b_args(s, seed=3):
    rng = np.random.default_rng(seed)
    return (s['S'], 1.7, s['F'], s['w'], np.zeros(s['rfft_shape']),
            rng.normal(size=s['f_shape']), s['evecs'],
            rng.normal(size=s['shape']), rng.normal(size=s['rfft_len']),
            rng.normal(size=s['f_shape']),
            rng.normal(size=int(np.prod(s['shape']))), s['shape'])


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------

def test_beam_is_symmetric():
    """``<Bx, y> == <x, By>``. The adjoint of ``B Us`` is taken to be
    ``Us^T B`` everywhere, which is only true if this holds."""
    beam = chromatic()
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(2, int(np.prod(SHAPE))))
    assert np.isclose(beam.apply(x) @ y, x @ beam.apply(y), rtol=1e-12)


def test_beam_conserves_the_mean_in_every_channel():
    beam = chromatic()
    cube = np.random.default_rng(2).normal(size=SHAPE)
    assert np.allclose(beam.apply(cube).mean(axis=(0, 1)),
                       cube.mean(axis=(0, 1)), rtol=1e-12, atol=1e-14)


def test_beam_keeps_the_layout_it_was_given():
    beam = chromatic()
    cube = np.random.default_rng(3).normal(size=SHAPE)
    assert beam.apply(cube).shape == SHAPE
    assert np.allclose(beam.apply(cube.ravel()), beam.apply(cube).ravel())


def test_beam_matches_a_directly_sampled_gaussian():
    """Independent of the FFT implementation: convolve a delta and compare
    with a periodic Gaussian sampled in real space. On a grid >= 11 sigma wide
    the two differ only by the periodic wrap, ~exp(-5.7**2 / 2) ~ 1e-7 on
    the short axis."""
    shape = (64, 48, 2)
    freqs = np.array([1000.0, 1100.0])
    fwhm = 3.0                                     # deg, at 1000 MHz
    beam = gaussian_beam(freqs, shape, fwhm_ref=fwhm, nu_ref=1000.0,
                         index=1.0, pix_deg=PIX)
    delta = np.zeros(shape)
    delta[0, 0, :] = 1.0
    out = beam.apply(delta)

    x = np.minimum(np.arange(64), 64 - np.arange(64)) * PIX
    y = np.minimum(np.arange(48), 48 - np.arange(48)) * PIX
    r2 = x[:, None] ** 2 + y[None, :] ** 2
    for c, nu in enumerate(freqs):
        sigma = fwhm * (1000.0 / nu) / FWHM_TO_SIGMA
        g = np.exp(-0.5 * r2 / sigma ** 2)
        g /= g.sum()
        assert np.abs(out[:, :, c] - g).max() < 1e-6 * g.max()


def test_fwhm_scales_as_one_over_nu_on_the_real_band():
    """Diffraction scaling across the live band: the 5.4% quoted in STATUS."""
    f = beam_fwhm(REAL_FREQS, 1.55, REAL_FREQS.mean())
    assert np.isclose(f[0] / f[-1], REAL_FREQS[-1] / REAL_FREQS[0])
    assert 0.053 < f[0] / f[-1] - 1 < 0.055


def test_full_transfer_is_the_half_plane_mirrored():
    """``full_transfer`` feeds both preconditioner blocks. Applying it with a
    full complex FFT must give the same answer as ``apply``."""
    beam = gaussian_beam(FREQS, (7, 5, 16), 0.6, FREQS[0], 1.0, pix_deg=PIX)
    cube = np.random.default_rng(4).normal(size=(7, 5, 16))
    via_full = np.fft.ifft2(np.fft.fft2(cube, axes=(0, 1))
                            * beam.full_transfer(), axes=(0, 1)).real
    assert np.allclose(via_full, beam.apply(cube), rtol=1e-12, atol=1e-14)


def test_beam_rejects_a_mismatched_transfer():
    with pytest.raises(ValueError):
        Beam(np.ones((8, 6, 16)), SHAPE)             # full plane, not half
    with pytest.raises(ValueError):
        gaussian_beam(FREQS[:-1], SHAPE, 1.0, 1000.0)


# ---------------------------------------------------------------------------
# The linear system
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('with_sys', [False, True])
def test_no_beam_leaves_the_system_untouched(system, with_sys):
    """``beam=None`` is the published operator term for term."""
    s = system
    kw = dict(sys_basis=s['basis'], G=s['G']) if with_sys else {}
    n = 2 * s['rfft_len'] + s['f_len'] + (s['basis'].n_params if with_sys
                                          else 0)
    x = np.random.default_rng(5).normal(size=n)
    assert np.array_equal(A_of(s, x, **kw), A_of(s, x, beam=None, **kw))
    assert np.array_equal(construct_b(*b_args(s), **kw),
                          construct_b(*b_args(s), beam=None, **kw))


@pytest.mark.parametrize('with_sys', [False, True])
def test_identity_beam_reproduces_the_beamless_system(system, with_sys):
    """The beam path is written as model-weight-adjoint rather than block by
    block. With ``B = 1`` the two must agree to rounding -- this is what ties
    the new path to the one the paper was run with."""
    s = system
    kw = (dict(sys_basis=s['basis'], G=s['G'], wg=None) if with_sys else {})
    kwA = {k: v for k, v in kw.items() if k != 'wg'}
    n = 2 * s['rfft_len'] + s['f_len'] + (s['basis'].n_params if with_sys
                                          else 0)
    x = np.random.default_rng(6).normal(size=n)
    one = identity_beam()
    assert np.allclose(A_of(s, x, beam=one, **kwA), A_of(s, x, **kwA),
                       rtol=1e-12, atol=1e-12)
    assert np.allclose(construct_b(*b_args(s), beam=one, **kw),
                       construct_b(*b_args(s), **kw), rtol=1e-12, atol=1e-12)


def test_rows_equal_the_normal_equations_written_out(system):
    """Every row of ``Ax`` and ``b`` with a beam, rebuilt from

        d = w (B Us s + B Uf f + Ug g) + n

    using the same rfft convention the beamless operator uses (``rfftn`` as
    the pull-back for ``s``; see the Hermitian-convention test in
    ``test_systematics.py``)."""
    s = system
    sb, G, beam = s['basis'], s['G'], chromatic()
    rng = np.random.default_rng(7)
    sig = rng.normal(size=2 * s['rfft_len'])
    f = rng.normal(size=s['f_shape'])
    g = rng.normal(size=sb.g_shape)
    x = np.concatenate([sig, f.ravel(), g.ravel()])
    Ax = A_of(s, x, sys_basis=sb, G=G, beam=beam)

    sk = (sig[:s['rfft_len']] + 1j * sig[s['rfft_len']:]).reshape(s['rfft_shape'])
    s_cube = np.fft.irfftn(sk, norm='ortho').ravel()
    f_cube = (f.reshape(-1, N_MODES) @ s['evecs']).ravel()
    r = s['Nw_inv'] * (beam.apply(s_cube + f_cube) + sb.apply(g))
    rB = beam.apply(r).reshape(SHAPE)
    S_r = s['S'].reshape(SHAPE)[:, :, :SHAPE[2] // 2 + 1]

    row_s = sk / S_r + np.fft.rfftn(rB, norm='ortho')
    row_f = f / s['F'] + rB @ s['evecs'].T
    row_g = (1 / G).reshape(sb.g_shape) * g + sb.adjoint(r)
    expected = np.concatenate([row_s.real.ravel(), row_s.imag.ravel(),
                               row_f.ravel(), row_g.ravel()])
    assert np.allclose(Ax, expected, rtol=1e-10, atol=1e-12)

    args = b_args(s, seed=8)
    S, N_inv, F, w, s_mean, f_mean, evecs, data, ws, wf, wd, _ = args
    wg = rng.normal(size=sb.g_shape)
    b = construct_b(*args, sys_basis=sb, G=G, wg=wg, beam=beam)
    rd = N_inv * w * data.ravel() + np.sqrt(N_inv) * np.sqrt(w) * wd
    rdB = beam.apply(rd).reshape(SHAPE)
    b_s = (np.fft.rfftn(rdB, norm='ortho') + s_mean / S_r
           + (ws / np.sqrt(S_r.ravel())).reshape(s['rfft_shape']))
    b_f = rdB @ evecs.T + f_mean / F + wf / np.sqrt(F)
    b_g = sb.adjoint(rd) + (1 / np.sqrt(G)).reshape(sb.g_shape) * wg
    expected_b = np.concatenate([b_s.real.ravel(), b_s.imag.ravel(),
                                 b_f.ravel(), b_g.ravel()])
    assert np.allclose(b, expected_b, rtol=1e-10, atol=1e-12)


def test_beam_keeps_the_real_space_blocks_symmetric(system):
    """``f-f``, ``f-g``, ``g-g`` contain no rfft packing, so with a symmetric
    ``B`` they must stay symmetric. A non-symmetric ``B`` would show here."""
    s = system
    A = dense_A(s, sys_basis=s['basis'], G=s['G'], beam=chromatic())
    n_sf = 2 * s['rfft_len'] + s['f_len']
    f = slice(2 * s['rfft_len'], n_sf)
    g = slice(n_sf, n_sf + s['basis'].n_params)
    assert np.allclose(A[f, f], A[f, f].T, rtol=1e-10, atol=1e-12)
    assert np.allclose(A[f, g], A[g, f].T, rtol=1e-10, atol=1e-12)
    assert np.allclose(A[g, g], A[g, g].T, rtol=1e-10, atol=1e-12)


def test_preconditioner_foreground_block_is_exact(system):
    """For uniform noise ``Uf^T B^2 Uf`` is block-diagonal in ``k_perp``, so
    ``M11`` inverts the true ``f-f`` block, not an approximation of it."""
    s = system
    beam = chromatic()
    A = dense_A(s, beam=beam)
    f = slice(2 * s['rfft_len'], 2 * s['rfft_len'] + s['f_len'])
    M = construct_preconditioner(s['S'], 2.0, s['F'], s['evecs'],
                                 s['rfft_len'], s['rfft_shape'], s['f_len'],
                                 s['f_shape'], s['shape'], beam=beam)
    v = np.random.default_rng(9).normal(size=s['f_len'])
    x = np.zeros(A.shape[0])
    x[f] = v
    assert np.allclose(M(x)[f], np.linalg.solve(A[f, f], v), rtol=1e-10)


def test_preconditioner_signal_block_is_exact_for_an_achromatic_beam(system):
    """With no chromaticity ``Us^T B^2 Us`` is diagonal, so ``1/M00`` must
    equal the diagonal of ``A`` on the interior ``kz`` modes (the ``kz = 0``
    and Nyquist planes carry the Hermitian packing and are excluded)."""
    s = system
    beam = chromatic(index=0.0)
    A = dense_A(s, beam=beam)
    M = construct_preconditioner(s['S'], 2.0, s['F'], s['evecs'],
                                 s['rfft_len'], s['rfft_shape'], s['f_len'],
                                 s['f_shape'], s['shape'], beam=beam)
    n = A.shape[0]
    m00 = 1 / M(np.ones(n))[:s['rfft_len']].reshape(s['rfft_shape'])
    diag = np.diag(A)[:s['rfft_len']].reshape(s['rfft_shape'])
    interior = (slice(None), slice(None), slice(1, SHAPE[2] // 2))
    assert np.allclose(m00[interior], diag[interior], rtol=1e-10)


def test_beamed_map_solve_recovers_the_foreground_that_ignoring_it_cannot(
        system):
    """A small end-to-end MAP solve on beamed data.

    Fit quality cannot tell the two models apart -- without a beam, ``s`` is
    free in every voxel and absorbs whatever ``Uf f`` misses -- so this
    compares against the truth instead. The foreground amplitudes are the
    clean discriminator on a grid this small: ignoring the beam recovers the
    *smoothed* amplitudes and gets them wrong by ~85%; modelling it gets
    within ~15%, the rest being transverse modes the beam has suppressed
    below the prior. LGMRES must also converge with the beam-aware
    preconditioner."""
    s = system
    beam = chromatic(fwhm=0.6)
    rng = np.random.default_rng(10)
    nvox = int(np.prod(SHAPE))
    fg = (rng.normal(size=(8, 6, N_MODES)) * 10) @ s['evecs']
    sky = rng.normal(size=SHAPE) * 0.1 + fg
    sigma = 0.01
    data = beam.apply(sky) + sigma * rng.normal(size=SHAPE)
    N_inv = 1 / sigma ** 2
    Nw_inv = N_inv * s['w']
    S = np.full(nvox, 0.01)
    F = np.full(N_MODES, 100.0)
    n = 2 * s['rfft_len'] + s['f_len']

    def solve(b_):
        kw = dict(beam=b_)
        L = LinearOperator((n, n), matvec=lambda v: construct_A(
            v, S, Nw_inv, F, s['w'], s['evecs'], s['rfft_len'],
            s['rfft_shape'], s['f_len'], s['f_shape'], SHAPE, **kw))
        M = LinearOperator((n, n), matvec=construct_preconditioner(
            S, N_inv, F, s['evecs'], s['rfft_len'], s['rfft_shape'],
            s['f_len'], s['f_shape'], SHAPE, **kw))
        b = construct_b(S, N_inv, F, s['w'], np.zeros(s['rfft_shape']),
                        np.zeros(s['f_shape']), s['evecs'], data,
                        np.zeros(s['rfft_len']), np.zeros(s['f_shape']),
                        np.zeros(nvox), SHAPE, **kw)
        x, info = lgmres(L, b, rtol=1e-10, atol=0, M=M, maxiter=500)
        assert info == 0
        return x[2 * s['rfft_len']:].reshape(s['f_shape']) @ s['evecs']

    err = {k: np.linalg.norm(solve(b_) - fg) / np.linalg.norm(fg)
           for k, b_ in (('beam', beam), ('none', None))}
    assert err['beam'] < 0.2
    assert err['none'] > 0.6


# ---------------------------------------------------------------------------
# The physics: chromatic mode mixing
# ---------------------------------------------------------------------------

def smooth_sky(seed=11, shape=SHAPE, n_modes=N_MODES):
    evecs = legendre_basis(shape[2], n_modes)
    amps = np.random.default_rng(seed).normal(size=shape[:2] + (n_modes,))
    return amps @ evecs, evecs


def test_achromatic_beam_adds_no_spectral_structure():
    """Smoothing alone commutes with the per-pixel spectral projection."""
    sky, evecs = smooth_sky()
    _, rms = chromatic_leakage(sky, chromatic(index=0.0), evecs)
    assert rms < 1e-12 * np.std(sky)


def test_chromatic_beam_pushes_a_smooth_sky_out_of_the_foreground_span():
    """The reason the beam is in the model at all."""
    sky, evecs = smooth_sky()
    _, rms = chromatic_leakage(sky, chromatic(index=1.0), evecs)
    assert rms > 1e-4 * np.std(sky)


def test_leakage_grows_with_chromaticity():
    sky, evecs = smooth_sky()
    rms = [chromatic_leakage(sky, chromatic(index=a), evecs)[1]
           for a in (0.5, 1.0, 2.0)]
    assert rms[0] < rms[1] < rms[2]


def test_uniform_sky_is_untouched_by_a_chromatic_beam():
    """``B(k_perp = 0) = 1`` in every channel: no spatial gradient, nothing
    for the chromaticity to convert."""
    evecs = legendre_basis(SHAPE[2], N_MODES)
    sky = np.broadcast_to(np.array([3.0, -1.0, 0.5]) @ evecs, SHAPE)
    _, rms = chromatic_leakage(sky, chromatic(index=2.0), evecs)
    assert rms < 1e-12


def test_leakage_rejects_a_non_orthonormal_basis():
    sky, evecs = smooth_sky()
    with pytest.raises(ValueError):
        chromatic_leakage(sky, chromatic(), 2 * evecs)


# ---------------------------------------------------------------------------
# Measured beams: tabulation and the collapsed S step
# ---------------------------------------------------------------------------

def test_hankel_transform_of_a_gaussian_profile_is_gaussian():
    """``radial_transfer`` against the closed form ``exp(-k^2 sigma^2 / 2)``."""
    from imgibbs import radial_transfer
    sigma = 0.6
    r = np.linspace(0, 8 * sigma, 4000)
    k = np.linspace(0, 6, 50)
    B = radial_transfer(r, np.exp(-0.5 * r ** 2 / sigma ** 2), k)
    assert np.allclose(B, np.exp(-0.5 * k ** 2 * sigma ** 2), atol=1e-6)


def test_tabulated_gaussian_reproduces_gaussian_beam():
    """A table built from a chromatic Gaussian must give back
    ``gaussian_beam`` -- the path a measured beam takes into the model."""
    from imgibbs import tabulated_beam
    nu_table = np.linspace(FREQS[0] - 1, FREQS[-1] + 1, 400)
    k = np.linspace(0, 20, 4001)
    sig = beam_fwhm(nu_table, 0.6, FREQS[0], 1.0) / FWHM_TO_SIGMA
    table = np.exp(-0.5 * k[None, :] ** 2 * sig[:, None] ** 2)
    tab = tabulated_beam(k, nu_table, table, FREQS, SHAPE, pix_deg=PIX)
    assert np.allclose(tab.transfer, chromatic().transfer, atol=1e-4)


def test_width_scale_stretches_a_tabulated_beam():
    """``width_scale`` stretches the beam in angle: a 2% wider table of a
    Gaussian is the Gaussian with a 2% wider FWHM."""
    from imgibbs import tabulated_beam
    nu_table = np.linspace(FREQS[0] - 1, FREQS[-1] + 1, 400)
    k = np.linspace(0, 20, 4001)
    sig = beam_fwhm(nu_table, 0.6, FREQS[0], 1.0) / FWHM_TO_SIGMA
    table = np.exp(-0.5 * k[None, :] ** 2 * sig[:, None] ** 2)
    tab = tabulated_beam(k, nu_table, table, FREQS, SHAPE, pix_deg=PIX,
                         width_scale=1.02)
    assert np.allclose(tab.transfer, chromatic(fwhm=0.6 * 1.02).transfer,
                       atol=1e-4)
    assert not np.allclose(tab.transfer, chromatic().transfer, atol=1e-3)


def test_load_beam_table_reads_a_grid_beam(tmp_path):
    """A transfer saved on the map's own full grid loads back exactly, and
    planes at other frequencies are interpolated linearly."""
    from imgibbs import load_beam_table
    ref = chromatic()
    path = tmp_path / 'grid.npz'
    np.savez(path, beam=ref.full_transfer(), freqs=FREQS)
    assert np.array_equal(load_beam_table(path, FREQS, SHAPE).transfer,
                          ref.transfer)
    # Every other plane, with the last kept so the band is covered.
    keep = np.r_[np.arange(0, len(FREQS) - 1, 2), len(FREQS) - 1]
    np.savez(path, beam=ref.full_transfer()[:, :, keep], freqs=FREQS[keep])
    beam = load_beam_table(path, FREQS, SHAPE)
    assert np.allclose(beam.transfer[:, :, keep], ref.transfer[:, :, keep])
    mid = 1
    lin = 0.5 * (ref.transfer[:, :, 0] + ref.transfer[:, :, 2])
    assert np.allclose(beam.transfer[:, :, mid], lin)
    with pytest.raises(ValueError):
        load_beam_table(path, FREQS, SHAPE, width_scale=1.02)


def test_load_beam_table_reads_azimuthal_profiles(tmp_path):
    """A pickle of Gaussian power profiles gives back the Gaussian beam."""
    import pickle
    from imgibbs import load_beam_table
    fr = np.linspace(FREQS[0] - 2, FREQS[-1] + 2, 40)
    th = np.arange(0, 6.0001, 0.005)
    sig = beam_fwhm(fr, 0.6, FREQS[0], 1.0) / FWHM_TO_SIGMA
    P = np.exp(-0.5 * th[None] ** 2 / sig[:, None] ** 2)
    path = tmp_path / 'profiles.pkl'
    with open(path, 'wb') as fh:
        pickle.dump(dict(freq=fr, th=th, P_v_th=P, P_h_th=P), fh)
    beam = load_beam_table(path, FREQS, SHAPE, pix_deg=PIX)
    assert np.allclose(beam.transfer, chromatic().transfer, atol=2e-3)
    assert 'azimuthal profiles' in beam.source


def test_load_beam_table_refuses_a_jones_model(tmp_path):
    """A full Jones model (GBs) must go through build_beam_table.py."""
    from imgibbs import load_beam_table
    path = tmp_path / 'jones.npz'
    np.savez(path, beam=np.zeros((4, 1, 2, 3, 3), complex),
             margin_deg=np.zeros(3), freq_MHz=np.zeros(2))
    with pytest.raises(ValueError, match='Jones'):
        load_beam_table(path, FREQS, SHAPE)


def test_tabulated_beam_refuses_to_extrapolate():
    from imgibbs import tabulated_beam
    k = np.linspace(0, 20, 10)
    nu = np.array([FREQS[0] + 1, FREQS[-1]])
    with pytest.raises(ValueError):
        tabulated_beam(k, nu, np.ones((2, 10)), FREQS, SHAPE, pix_deg=PIX)
    nu = np.array([FREQS[0] - 1, FREQS[-1] + 1])
    with pytest.raises(ValueError):
        tabulated_beam(np.linspace(0, 1, 10), nu, np.ones((2, 10)), FREQS,
                       SHAPE, pix_deg=PIX)


def test_observable_modes_follow_the_band_averaged_beam():
    """Visible modes are the low ``k_perp`` ones, constant along ``k_par``,
    laid out like ``idxs`` (full fftn grid, flattened)."""
    beam = chromatic(fwhm=0.9)
    vis = beam.observable_modes(1e-2).reshape(SHAPE)
    assert vis[0, 0].all()                        # k_perp = 0 always visible
    assert (vis == vis[:, :, :1]).all()           # same at every k_par
    b2 = (beam.full_transfer() ** 2).mean(axis=2)
    assert np.array_equal(vis[:, :, 0], b2 >= 1e-2)
    assert 0 < vis.mean() < 1


# ---------------------------------------------------------------------------
# The exact-mask foreground preconditioner
# ---------------------------------------------------------------------------

def masked_system(system, seed=12):
    """The small system with a real mask: an empty corner and flagged channels."""
    s = dict(system)
    rng = np.random.default_rng(seed)
    w = (rng.random(SHAPE) > 0.15).astype(float)
    w[:3, :2, :] = 0.0                                  # pixels with no data
    s['w'] = w.ravel()
    s['Nw_inv'] = 2.0 * s['w']
    return s


def test_masked_block_with_every_channel_is_the_exact_inverse(system):
    """One beam group per channel and every mode dense: the block must invert
    the true masked ``f-f`` block exactly, chromaticity and all."""
    from imgibbs import MaskedForegroundBlock
    s = masked_system(system)
    beam = chromatic()
    A = dense_A(s, beam=beam)
    f = slice(2 * s['rfft_len'], 2 * s['rfft_len'] + s['f_len'])
    blk = MaskedForegroundBlock(beam, s['w'], s['evecs'], 2.0,
                                n_dense=N_MODES, n_groups=SHAPE[2])
    blk.factor(s['F'])
    M = construct_preconditioner(s['S'], 2.0, s['F'], s['evecs'],
                                 s['rfft_len'], s['rfft_shape'], s['f_len'],
                                 s['f_shape'], s['shape'], beam=beam,
                                 fg_block=blk)
    v = np.random.default_rng(13).normal(size=s['f_len'])
    x = np.zeros(A.shape[0])
    x[f] = v
    assert np.allclose(M(x)[f], np.linalg.solve(A[f, f], v), rtol=1e-8)


def test_masked_block_leaves_the_other_blocks_alone(system):
    from imgibbs import MaskedForegroundBlock
    s = masked_system(system)
    beam = chromatic()
    blk = MaskedForegroundBlock(beam, s['w'], s['evecs'], 2.0, n_dense=2)
    blk.factor(s['F'])
    common = (s['S'], 2.0, s['F'], s['evecs'], s['rfft_len'],
              s['rfft_shape'], s['f_len'], s['f_shape'], s['shape'])
    old = construct_preconditioner(*common, beam=beam)
    new = construct_preconditioner(*common, beam=beam, fg_block=blk)
    x = np.random.default_rng(14).normal(size=2 * s['rfft_len'] + s['f_len'])
    sig = slice(0, 2 * s['rfft_len'])
    f = slice(2 * s['rfft_len'], None)
    assert np.array_equal(old(x)[sig], new(x)[sig])
    # modes >= n_dense still use the Fourier block, over those modes only
    assert not np.allclose(old(x)[f], new(x)[f])


def test_masked_block_rescues_a_stiff_masked_solve(system):
    """The live-grid failure in miniature: a weak foreground prior, a mask and
    a beam. The uniform-coverage preconditioner stalls; the exact-mask block
    converges in a handful of iterations."""
    from imgibbs import MaskedForegroundBlock
    s = masked_system(system)
    beam = chromatic(fwhm=0.9)
    F = np.array([1e6, 1e2, 1e-2])                      # N^-1 F ~ 2e6 .. 2e-2
    f = slice(2 * s['rfft_len'], 2 * s['rfft_len'] + s['f_len'])
    b = np.random.default_rng(15).normal(size=s['f_len'])

    def iterations(M):
        cnt = [0]

        def mv(v):
            cnt[0] += 1
            x = np.zeros(2 * s['rfft_len'] + s['f_len'])
            x[f] = v
            return construct_A(x, s['S'], s['Nw_inv'], F, s['w'], s['evecs'],
                               s['rfft_len'], s['rfft_shape'], s['f_len'],
                               s['f_shape'], s['shape'], beam=beam)[f]

        def pc(v):
            x = np.zeros(2 * s['rfft_len'] + s['f_len'])
            x[f] = v
            return M(x)[f]
        n = s['f_len']
        lgmres(LinearOperator((n, n), matvec=mv), b, rtol=1e-8, atol=0,
               M=LinearOperator((n, n), matvec=pc), maxiter=40)
        return cnt[0]

    common = (s['S'], 2.0, F, s['evecs'], s['rfft_len'], s['rfft_shape'],
              s['f_len'], s['f_shape'], s['shape'])
    blk = MaskedForegroundBlock(beam, s['w'], s['evecs'], 2.0, n_dense=2,
                                n_groups=4)
    blk.factor(F)
    plain = iterations(construct_preconditioner(*common, beam=beam))
    exact = iterations(construct_preconditioner(*common, beam=beam,
                                                fg_block=blk))
    assert exact < 40
    assert exact < plain / 3


def test_beam_spec_round_trips_to_run_gibbs_flags():
    """beam_injection records a beam as a spec and hands run_gibbs the flags;
    the two must describe the same beam."""
    from imgibbs import beam_from_spec, beam_run_args
    spec = dict(fwhm=0.6, index=1.0, nu_ref=float(FREQS[0]))
    b = beam_from_spec(spec, FREQS, SHAPE, pix_deg=PIX)
    assert np.allclose(b.transfer, chromatic().transfer)
    flags = beam_run_args(spec)
    assert flags[flags.index('--beam-fwhm') + 1] == repr(0.6)
    assert flags[flags.index('--beam-nu-ref') + 1] == repr(float(FREQS[0]))
    assert beam_run_args(dict(table='t.npz')) == ['--beam-table', 't.npz']


# ---------------------------------------------------------------------------
# Projecting B Uf out of a systematic template (the smooth ground spill)
# ---------------------------------------------------------------------------

def smooth_basis():
    return groundspill_basis(FREQS, SHAPE, period=40.0, include_smooth=True)


def test_project_out_beamed_leaves_nothing_b_uf_can_fit():
    """The projected template has no part ``B Uf f`` can fit, and what was
    removed is entirely such a part. Together: the model's span is the same
    with the projected template as with the original."""
    beam, evecs = chromatic(index=2.0), legendre_basis(SHAPE[2], N_MODES)
    t = smooth_basis().cube(np.eye(1, smooth_basis().n_params).reshape(
        smooth_basis().g_shape))
    proj = project_out_beamed(t, beam.full_transfer(), evecs)
    again = project_out_beamed(proj, beam.full_transfer(), evecs)
    removed = project_out_beamed(t - proj, beam.full_transfer(), evecs)
    assert np.abs(again - proj).max() < 1e-12 * np.abs(t).max()
    assert np.abs(removed).max() < 1e-12 * np.abs(t).max()
    # A chromatic beam makes the smooth template only nearly reproducible.
    assert 0 < np.abs(proj).max() < 0.5 * np.abs(t).max()


def test_project_out_beamed_skips_beam_killed_modes():
    """Where the beam is zero, ``B Uf f`` fits nothing, so nothing may be
    projected out there -- an arbitrary basis for the null space would be."""
    evecs = legendre_basis(SHAPE[2], N_MODES)
    transfer = np.zeros(SHAPE)
    transfer[0, 0] = 1.0                                     # only k_perp = 0
    t = np.random.default_rng(3).normal(size=SHAPE)
    proj = project_out_beamed(t, transfer, evecs)
    ft, fp = (np.fft.fft2(a, axes=(0, 1)) for a in (t, proj))
    assert np.allclose(fp[1:], ft[1:]) and np.allclose(fp[0, 1:], ft[0, 1:])
    assert np.allclose(fp[0, 0] @ evecs.T, 0, atol=1e-10)


def test_beam_project_keeps_unselected_templates_and_is_a_true_basis():
    beam, evecs = chromatic(index=2.0), legendre_basis(SHAPE[2], N_MODES)
    sep = smooth_basis()
    tb = beam_project(sep, beam.full_transfer(), evecs, rows=[0])
    assert tb.g_shape == sep.g_shape and tb.n_params == sep.n_params
    rng = np.random.default_rng(4)
    g = rng.normal(size=sep.g_shape)
    g_ripple = g.copy()
    g_ripple[:, 0] = 0
    assert np.allclose(tb.apply(g_ripple), sep.apply(g_ripple))
    y = rng.normal(size=int(np.prod(SHAPE)))
    assert np.isclose(tb.apply(g) @ y, (g * tb.adjoint(y)).sum(), rtol=1e-12)
    explicit = np.array([tb.apply(e) for e in np.eye(tb.n_params)])
    assert np.allclose(tb.gram(), explicit @ explicit.T)


# ---------------------------------------------------------------------------
# Integrating the systematic out (MarginalNoise)
# ---------------------------------------------------------------------------

def masked_weights(seed=5):
    """``w N^-1`` with some voxels flagged."""
    w = (np.random.default_rng(seed).random(int(np.prod(SHAPE))) > 0.2)
    return 2.0 * w


def test_marginal_noise_is_the_woodbury_inverse():
    """Unmasked, ``N'^-1`` must equal the inverse of ``N + Ug G Ug^T``."""
    basis = smooth_basis()
    rng = np.random.default_rng(6)
    W = rng.uniform(1.0, 3.0, int(np.prod(SHAPE)))
    G = rng.uniform(0.5, 2.0, basis.n_params)
    mn = MarginalNoise(W, basis, G)
    U = np.array([basis.apply(e) for e in np.eye(basis.n_params)]).T
    dense = np.linalg.inv(np.diag(1 / W) + U @ np.diag(G) @ U.T)
    y = rng.normal(size=W.size)
    assert np.allclose(mn * y, dense @ y, rtol=1e-9)
    assert np.allclose(y * mn, mn * y)            # numpy must not broadcast


def test_marginal_weighted_data_has_the_right_mean_and_covariance():
    """``weighted_data`` is linear in the draws; its covariance must be
    ``N'^-1`` exactly, masked voxels included, and its mean ``N'^-1 d``."""
    basis = smooth_basis()
    W = masked_weights()
    G = np.random.default_rng(7).uniform(0.5, 2.0, basis.n_params)
    mn = MarginalNoise(W, basis, G)
    n, p = W.size, basis.n_params
    d = np.random.default_rng(8).normal(size=n)
    zero_d, zero_g = np.zeros(n), np.zeros(p)
    assert np.allclose(mn.weighted_data(d, zero_d, zero_g), mn * d)
    J = np.array([mn.weighted_data(zero_d, e, zero_g) for e in np.eye(n)]
                 + [mn.weighted_data(zero_d, zero_d, e) for e in np.eye(p)]).T
    dense = np.array([mn * e for e in np.eye(n)]).T
    assert np.allclose(J @ J.T, dense, atol=1e-12)


def test_marginal_solve_matches_the_joint_solve():
    """MAP solution (omegas zero), masked and beamed: solving for (s, f) with
    ``g`` integrated out and then taking ``E[g | s, f, d]`` must give the
    joint solve's (s, f, g) exactly. This is the whole claim."""
    nx, ny, nz = SHAPE
    rng = np.random.default_rng(9)
    beam = chromatic(index=2.0)
    basis = smooth_basis()
    evecs = legendre_basis(nz, N_MODES)
    S = np.abs(rng.normal(size=nx * ny * nz)) + 1.0
    F = np.abs(rng.normal(size=N_MODES)) + 1.0
    G = rng.uniform(0.5, 2.0, basis.n_params)
    W = masked_weights()
    w = (W > 0).astype(float)
    rl, rs = nx * ny * (nz // 2 + 1), (nx, ny, nz // 2 + 1)
    fl, fs = nx * ny * N_MODES, (nx, ny, N_MODES)
    d = rng.normal(size=SHAPE) * w.reshape(SHAPE)
    zeros = dict(ws=np.zeros(rl), wf=np.zeros(fs), wd=np.zeros(W.size))
    common = (S, 2.0, F, w, np.zeros(rs), np.zeros(fs), evecs, d)

    def dense(Nw, sys_basis, n):
        return np.array([construct_A(np.eye(n)[i], S, Nw, F, w, evecs, rl, rs,
                                     fl, fs, SHAPE, sys_basis=sys_basis,
                                     G=G, beam=beam)
                         for i in range(n)]).T

    n_sf = 2 * rl + fl
    b_joint = construct_b(*common, zeros['ws'], zeros['wf'], zeros['wd'],
                          SHAPE, sys_basis=basis, G=G,
                          wg=np.zeros(basis.g_shape), beam=beam)
    x_joint = np.linalg.solve(dense(W, basis, n_sf + basis.n_params), b_joint)

    mn = MarginalNoise(W, basis, G)
    b_marg = construct_b(*common, zeros['ws'], zeros['wf'], zeros['wd'],
                         SHAPE, beam=beam,
                         weighted_data=mn.weighted_data(
                             d, zeros['wd'], np.zeros(basis.n_params)))
    x_marg = np.linalg.solve(dense(mn, None, n_sf), b_marg)
    s = Us((x_marg[:rl] + 1j * x_marg[rl:2 * rl]).reshape(rs), False)
    f = x_marg[2 * rl:].reshape(fs)
    sky = beam.apply(s.ravel() + Uf(evecs, f, False))
    g = mn.draw(d.ravel() - sky, np.zeros(basis.n_params))

    assert np.allclose(x_marg, x_joint[:n_sf], rtol=1e-8, atol=1e-10)
    assert np.allclose(g.ravel(), x_joint[n_sf:], rtol=1e-8, atol=1e-10)


def test_marginal_preconditioner_is_exact_given_the_exact_A0_inverse():
    """Woodbury on the preconditioner: with ``M1 = A0^-1`` (plain ``W``
    weights), the corrected operator must be the exact inverse of the
    marginal system."""
    nx, ny, nz = SHAPE
    rng = np.random.default_rng(10)
    beam = chromatic(index=2.0)
    basis = smooth_basis()
    evecs = legendre_basis(nz, N_MODES)
    S = np.abs(rng.normal(size=nx * ny * nz)) + 1.0
    F = np.abs(rng.normal(size=N_MODES)) + 1.0
    G = rng.uniform(0.5, 2.0, basis.n_params)
    W = masked_weights()
    w = (W > 0).astype(float)
    rl, rs = nx * ny * (nz // 2 + 1), (nx, ny, nz // 2 + 1)
    fl, fs = nx * ny * N_MODES, (nx, ny, N_MODES)
    n = 2 * rl + fl

    def dense(Nw):
        return np.array([construct_A(np.eye(n)[i], S, Nw, F, w, evecs, rl, rs,
                                     fl, fs, SHAPE, beam=beam)
                         for i in range(n)]).T

    mn = MarginalNoise(W, basis, G)
    A0_inv = np.linalg.inv(dense(W))
    C = np.array([construct_b(S, 2.0, F, w, np.zeros(rs), np.zeros(fs),
                              evecs, np.zeros(SHAPE), np.zeros(rl),
                              np.zeros(fs), np.zeros(W.size), SHAPE,
                              beam=beam, weighted_data=row)
                  for row in mn.WU])
    M = mn.preconditioner(lambda r: A0_inv @ r, C, rfft_adjoint_weight(rs, fl))
    r = rng.normal(size=n)
    assert np.allclose(M(r), np.linalg.solve(dense(mn), r), rtol=1e-7)
