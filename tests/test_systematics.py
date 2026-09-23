"""Tests for the systematics block.

Three things here are worth more than the rest.

``test_groundspill_block_matches_the_existing_hermitian_convention`` pins the
new signal-systematic coupling to the same convention the existing
signal-foreground coupling already uses. ``construct_A`` is not a symmetric
operator -- the blocks coupling ``s`` to anything in real space differ from
their transpose by the rfft Hermitian weight (see the test's docstring). That
predates this module; what matters is that the systematic block does not
introduce a *second*, different convention.

``test_smooth_spill_is_entirely_inside_the_foreground_basis`` and
``test_ripple_survives_the_foreground_basis`` pin the measurement the whole
design rests on: the smooth part of ground spill cannot be separated from the
foreground and does not need to be, while the standing-wave ripple sails
through a 6-mode clean and lands in the lowest signal bins.

Run with:  pytest -q tests/test_systematics.py
"""

import numpy as np
import pytest

from imgibbs import (
    RM_DEFAULT, SystematicBasis, best_fit_amplitudes, construct_A, construct_b,
    construct_preconditioner, faraday_templates, groundspill_basis,
    groundspill_cube, lambda_squared, leakage_basis, onef_basis,
    onef_covariance, period_scan, poly2d_templates, realise,
    ripple_wavenumber, scan_templates, spectral_templates, spillover_envelope,
)

# A small grid, so the dense-matrix tests below are cheap. The band is
# stretched to 15x the real channel width so a realistic ripple period still
# fits several cycles into 16 channels.
SHAPE = (8, 6, 16)
FREQS = 970.94 + np.arange(SHAPE[2]) * 0.208984 * 15
N_MODES = 3

# The real grid, for the degeneracy measurements that must be quoted on it.
REAL_FREQS = 970.94140625 + np.arange(250) * (1712.0 - 856.0) / 4096
REAL_LZ = 254.39405269274425


def legendre_basis(n_freq, n_modes):
    """The sampler's foreground basis: orthonormalised Legendre polynomials."""
    vander = np.polynomial.legendre.legvander(np.linspace(-1, 1, n_freq),
                                              n_modes - 1)
    return np.linalg.qr(vander)[0].T


@pytest.fixture(scope='module')
def basis():
    return groundspill_basis(FREQS, SHAPE, period=40.0)


@pytest.fixture(scope='module')
def system(basis):
    """Everything ``construct_A`` needs, on the small grid."""
    nx, ny, nz = SHAPE
    rng = np.random.default_rng(0)
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


def dense_A(system, with_sys=True):
    """Form ``A`` column by column. Only tractable because SHAPE is small."""
    s = system
    sb = s['basis'] if with_sys else None
    G = s['G'] if with_sys else None
    n = 2 * s['rfft_len'] + s['f_len'] + (sb.n_params if with_sys else 0)
    cols = [construct_A(np.eye(n)[i], s['S'], s['Nw_inv'], s['F'], s['w'],
                        s['evecs'], s['rfft_len'], s['rfft_shape'], s['f_len'],
                        s['f_shape'], s['shape'], sys_basis=sb, G=G)
            for i in range(n)]
    return np.array(cols).T


# ---------------------------------------------------------------------------
# The basis operator
# ---------------------------------------------------------------------------

def test_apply_and_adjoint_are_a_true_adjoint_pair(basis):
    """``<Ug g, y> == <g, Ug^T y>``. If this fails the solve is not symmetric
    in the systematic block and the posterior is not the one advertised."""
    rng = np.random.default_rng(1)
    g = rng.normal(size=basis.g_shape)
    y = rng.normal(size=int(np.prod(SHAPE)))
    lhs = float(basis.apply(g) @ y)
    rhs = float((g * basis.adjoint(y)).sum())
    assert np.isclose(lhs, rhs, rtol=1e-12)


def test_gram_factorisation_is_exact(basis):
    """``gram()`` uses a Kronecker product instead of assembling ``Ug^T Ug``
    over every voxel. That is an identity for separable templates, not an
    approximation -- so it must match the explicit product exactly."""
    eye = np.eye(basis.n_params)
    explicit = np.array([basis.apply(eye[i]) for i in range(basis.n_params)])
    assert np.allclose(explicit @ explicit.T, basis.gram(), rtol=1e-12)


def test_templates_are_unit_rms(basis):
    """Amplitudes in ``g`` are then read directly in K, which is what makes a
    prior width for ``G`` something you can reason about."""
    assert np.allclose(np.sqrt(np.mean(basis.spatial ** 2, axis=1)), 1.0)
    assert np.allclose(np.sqrt(np.mean(basis.spectral ** 2, axis=1)), 1.0)


def test_basis_rejects_a_mismatched_cube():
    with pytest.raises(ValueError, match='pixels'):
        SystematicBasis(spatial=np.ones((2, 7)),
                        spectral=np.ones((2, SHAPE[2])), shape=SHAPE)
    with pytest.raises(ValueError, match='channels'):
        SystematicBasis(spatial=np.ones((2, SHAPE[0] * SHAPE[1])),
                        spectral=np.ones((2, 5)), shape=SHAPE)


def test_scan_templates_vary_only_along_the_scan_axis():
    """Ground pickup is fixed in the telescope frame, so it must be constant
    across the axis the telescope did not scan in."""
    for axis in (0, 1):
        tmpl = scan_templates(SHAPE, order=1, scan_axis=axis)
        gradient = tmpl[1].reshape(SHAPE[0], SHAPE[1])
        other = 1 - axis
        assert np.allclose(np.ptp(gradient, axis=other), 0.0)
        assert np.ptp(gradient, axis=axis).max() > 0.1


# ---------------------------------------------------------------------------
# The degeneracy the design rests on
# ---------------------------------------------------------------------------

def test_smooth_spill_is_entirely_inside_the_foreground_basis():
    """The smooth spillover envelope is absorbed to machine precision at every
    ``n_modes``, so it is neither identifiable nor harmful. This is why
    :func:`spectral_templates` leaves it out by default -- putting it in ``Ug``
    would add a direction the data cannot constrain at all."""
    env = spillover_envelope(REAL_FREQS)
    env = env / np.linalg.norm(env)
    for n_modes in (6, 10, 20):
        fg = legendre_basis(len(REAL_FREQS), n_modes)
        absorbed = float(np.sum((fg @ env) ** 2))
        assert absorbed > 1 - 1e-9, f'n_modes={n_modes}: {absorbed}'


def test_ripple_survives_the_foreground_basis():
    """A standing wave with a few cycles across the band passes through the
    6-mode clean the sampler actually runs. The numbers are the table in
    ``imgibbs/systematics.py``; if they move, that table is wrong."""
    fg6 = legendre_basis(len(REAL_FREQS), 6)
    absorbed = period_scan(REAL_FREQS, [40.0, 20.0, 17.5, 10.0, 5.0], fg6)
    assert np.allclose(absorbed, [0.9945, 0.2872, 0.1837, 0.0577, 0.0149],
                       atol=5e-4)
    # Monotone: the shorter the period, the less the foreground can take.
    assert np.all(np.diff(absorbed) < 0)


def test_more_foreground_modes_do_absorb_the_ripple():
    """The reason this is not the fix: ``docs/STATUS.md`` records that
    ``n_modes = 20`` removes the 21cm signal too (T(k) 0.006-0.06) and pushes
    tau_int to 54-105. The absorption is real, the cure is worse."""
    absorbed = [float(period_scan(REAL_FREQS, [17.5],
                                  legendre_basis(len(REAL_FREQS), n))[0])
                for n in (6, 10, 20)]
    assert absorbed[0] < 0.2 < absorbed[1] < absorbed[2]
    assert absorbed[2] > 0.999


def test_period_scan_rejects_a_non_orthonormal_basis():
    raw = np.polynomial.legendre.legvander(
        np.linspace(-1, 1, len(REAL_FREQS)), 5).T
    with pytest.raises(ValueError, match='orthonormal'):
        period_scan(REAL_FREQS, [17.5], raw)


def test_ripple_lands_in_the_low_signal_bins():
    """A ripple is a single k_parallel mode, so all of its power goes into one
    bin. On the live grid the 10-20 MHz range covers the two lowest bin
    centres, 0.068 and 0.160 Mpc^-1 -- exactly where the 21cm measurement is."""
    bandwidth = REAL_FREQS[-1] - REAL_FREQS[0]
    k = ripple_wavenumber(np.array([20.0, 17.5, 10.0]), bandwidth, REAL_LZ)
    assert np.allclose(k, [0.0643, 0.0734, 0.1285], atol=1e-3)
    assert k[0] < 0.16 and k[-1] < 0.16      # all inside the lowest two bins


# ---------------------------------------------------------------------------
# The linear system
# ---------------------------------------------------------------------------

def test_no_basis_leaves_the_three_block_system_untouched(system):
    """``sys_basis=None`` must reproduce the old operator term for term, not
    just closely -- these tests would otherwise silently re-baseline the
    published sampler."""
    s = system
    rng = np.random.default_rng(2)
    n_old = 2 * s['rfft_len'] + s['f_len']
    x_old = rng.normal(size=n_old)
    x_new = np.concatenate([x_old, np.zeros(s['basis'].n_params)])

    old = construct_A(x_old, s['S'], s['Nw_inv'], s['F'], s['w'], s['evecs'],
                      s['rfft_len'], s['rfft_shape'], s['f_len'], s['f_shape'],
                      s['shape'])
    new = construct_A(x_new, s['S'], s['Nw_inv'], s['F'], s['w'], s['evecs'],
                      s['rfft_len'], s['rfft_shape'], s['f_len'], s['f_shape'],
                      s['shape'], sys_basis=s['basis'], G=s['G'])
    # Exact equality: with g = 0 the added terms are identically zero and are
    # added last, so no rounding can creep into the first three blocks.
    assert np.array_equal(old, new[:n_old])
    # The g row itself is NOT zero at g = 0 -- it still carries the signal and
    # foreground coupling, which is the whole point of solving jointly.
    assert len(new) == n_old + s['basis'].n_params
    assert np.abs(new[n_old:]).max() > 0


def test_b_with_no_basis_is_unchanged(system):
    s = system
    rng = np.random.default_rng(3)
    data = rng.normal(size=s['shape'])
    s_mean = np.zeros(s['rfft_shape'])
    f_mean = rng.normal(size=s['f_shape'])
    ws = rng.normal(size=s['rfft_len'])
    wf = rng.normal(size=s['f_shape'])
    wd = rng.normal(size=int(np.prod(s['shape'])))

    args = (s['S'], 1.0, s['F'], s['w'], s_mean, f_mean, s['evecs'], data,
            ws, wf, wd, s['shape'])
    old = construct_b(*args)
    new = construct_b(*args, sys_basis=s['basis'], G=s['G'],
                      wg=np.zeros(s['basis'].g_shape))
    n_old = 2 * s['rfft_len'] + s['f_len']
    assert np.array_equal(old, new[:n_old])
    assert len(new) == n_old + s['basis'].n_params


def test_systematic_blocks_are_exactly_symmetric(system):
    """The foreground-systematic and systematic-systematic blocks are pure
    real-space operations, so they carry none of the rfft packing weight and
    must be symmetric to machine precision."""
    s = system
    A = dense_A(system)
    n_sf = 2 * s['rfft_len'] + s['f_len']
    f = slice(2 * s['rfft_len'], n_sf)
    g = slice(n_sf, n_sf + s['basis'].n_params)

    assert np.allclose(A[f, g], A[g, f].T, rtol=1e-10, atol=1e-12)
    assert np.allclose(A[g, g], A[g, g].T, rtol=1e-10, atol=1e-12)


def test_groundspill_block_matches_the_existing_hermitian_convention(system):
    """``A`` is NOT symmetric across the blocks that couple ``s`` to real
    space, and that predates this module.

    ``Us(..., True)`` is ``rfftn``, which is the pseudo-inverse of ``irfftn``
    rather than its transpose: ``irfftn`` sums each interior mode together
    with its conjugate, so the true adjoint is ``2 x rfftn`` there and
    ``1 x rfftn`` on the ``kz = 0`` and ``kz = Nz/2`` planes. The existing
    signal-foreground block inherits exactly that weight.

    This test does not judge that choice. It pins that the systematic block
    reproduces it, so there is one convention in the operator and not two.
    """
    s = system
    A = dense_A(system)
    n_sf = 2 * s['rfft_len'] + s['f_len']
    sig = slice(0, 2 * s['rfft_len'])
    f = slice(2 * s['rfft_len'], n_sf)
    g = slice(n_sf, n_sf + s['basis'].n_params)

    # Recover the per-signal-mode weight from the EXISTING foreground block.
    Asf, Afs = A[sig, f], A[f, sig].T
    num = np.einsum('ij,ij->i', Afs, Asf)
    den = np.einsum('ij,ij->i', Asf, Asf)
    live = den > 1e-12 * den.max()
    weight = np.where(live, num / np.where(live, den, 1.0), 1.0)

    # It really is the Hermitian weight: 1 or 2, nothing else.
    assert np.all(np.isin(np.round(weight[live], 6), [1.0, 2.0]))
    assert np.allclose(weight[:, None] * Asf, Afs, rtol=1e-8, atol=1e-10)

    # The systematic block must follow the SAME weight, not a second one.
    assert np.allclose(weight[:, None] * A[sig, g], A[g, sig].T,
                       rtol=1e-8, atol=1e-10)


def test_preconditioner_grows_and_leaves_the_old_blocks_alone(system):
    s = system
    rng = np.random.default_rng(4)
    n_old = 2 * s['rfft_len'] + s['f_len']
    common = (s['S'], 1.0, s['F'], s['evecs'], s['rfft_len'], s['rfft_shape'],
              s['f_len'], s['f_shape'], s['shape'])
    old = construct_preconditioner(*common)
    new = construct_preconditioner(*common, sys_basis=s['basis'], G=s['G'])

    x = rng.normal(size=n_old + s['basis'].n_params)
    assert np.array_equal(old(x[:n_old]), new(x)[:n_old])
    assert len(new(x)) == n_old + s['basis'].n_params


def test_preconditioner_approximates_the_systematic_block(system):
    """``M22`` should invert the true ``g-g`` block when the mask is uniform,
    which is the regime the scalar ``N_inv`` in the preconditioner assumes."""
    s = system
    A = dense_A(system)
    n_sf = 2 * s['rfft_len'] + s['f_len']
    g = slice(n_sf, n_sf + s['basis'].n_params)
    M22 = np.diag(1.0 / s['G']) + 2.0 * s['basis'].gram()   # Nw_inv is 2.0
    assert np.allclose(A[g, g], M22, rtol=1e-10)


def test_g_row_equals_the_normal_equations_written_out(system):
    """Rebuild the systematic row of ``Ax`` and of ``b`` from the definitions,
    independently of the implementation, and check they agree.

    This is the only test that covers ``b2``. It avoids the rfft packing
    question entirely, because the ``g`` row is a pure real-space projection:
    every term is ``Ug^T`` applied to something, plus the prior.
    """
    s = system
    sb, G = s['basis'], s['G']
    rng = np.random.default_rng(6)
    nvox = int(np.prod(s['shape']))

    sig = rng.normal(size=2 * s['rfft_len'])
    f = rng.normal(size=s['f_shape'])
    g = rng.normal(size=sb.g_shape)
    x = np.concatenate([sig, f.ravel(), g.ravel()])

    Ax = construct_A(x, s['S'], s['Nw_inv'], s['F'], s['w'], s['evecs'],
                     s['rfft_len'], s['rfft_shape'], s['f_len'], s['f_shape'],
                     s['shape'], sys_basis=sb, G=G)
    g_row = Ax[2 * s['rfft_len'] + s['f_len']:]

    # Ug^T Nw_inv (Us s + Uf f + Ug g) + G^-1 g, written out by hand.
    s_cube = np.fft.irfftn(
        (sig[:s['rfft_len']] + 1j * sig[s['rfft_len']:]).reshape(s['rfft_shape']),
        norm='ortho').ravel()
    f_cube = (f.reshape(-1, N_MODES) @ s['evecs']).ravel()
    expected = (sb.adjoint(s['Nw_inv'] * (s_cube + f_cube + sb.apply(g)))
                + (1 / G).reshape(sb.g_shape) * g)
    assert np.allclose(g_row, expected.ravel(), rtol=1e-10)

    # And the RHS: Ug^T (N^-1 w d + N^-1/2 w^1/2 wd) + G^-1 g_mean + G^-1/2 wg.
    data = rng.normal(size=s['shape'])
    wd = rng.normal(size=nvox)
    wg = rng.normal(size=sb.g_shape)
    g_mean = rng.normal(size=sb.g_shape)
    N_inv = 3.0

    b = construct_b(s['S'], N_inv, s['F'], s['w'], np.zeros(s['rfft_shape']),
                    f, s['evecs'], data, rng.normal(size=s['rfft_len']),
                    rng.normal(size=s['f_shape']), wd, s['shape'],
                    sys_basis=sb, G=G, wg=wg, g_mean=g_mean)
    b2 = b[2 * s['rfft_len'] + s['f_len']:]

    expected_b = (sb.adjoint(N_inv * s['w'] * data.ravel()
                             + np.sqrt(N_inv) * np.sqrt(s['w']) * wd)
                  + (1 / G).reshape(sb.g_shape) * g_mean
                  + (1 / np.sqrt(G)).reshape(sb.g_shape) * wg)
    assert np.allclose(b2, expected_b.ravel(), rtol=1e-10)


# ---------------------------------------------------------------------------
# Injection and recovery
# ---------------------------------------------------------------------------

def test_injected_cube_has_the_requested_rms():
    cube, truth = groundspill_cube(FREQS, SHAPE, ripple_rms=1e-3)
    assert np.isclose(cube.std(), 1e-3, rtol=1e-10)
    assert np.allclose(truth['basis'].cube(truth['g_true']), cube, atol=1e-15)


def test_smooth_spill_does_not_enter_the_recorded_truth():
    """``spill_level`` adds a component that is deliberately NOT in
    ``truth['basis']`` -- it is there to be absorbed by the foreground, and a
    recovery test must not be scored against it."""
    cube, truth = groundspill_cube(FREQS, SHAPE, ripple_rms=1e-3,
                                   spill_level=0.5)
    assert cube.std() > 0.4
    ripple_only = truth['basis'].cube(truth['g_true'])
    assert np.isclose(ripple_only.std(), 1e-3, rtol=1e-10)


def test_uniform_spill_is_still_separable_from_the_foreground():
    """``gradient=0`` is the pure drift-scan limit: the contaminant is the
    same in every pixel. It is still recoverable, because what separates it
    from the foreground is its spectrum, not its spatial structure."""
    cube, truth = groundspill_cube(FREQS, SHAPE, ripple_rms=1e-3, gradient=0.0)
    per_pixel = cube.reshape(-1, SHAPE[2])
    assert np.allclose(per_pixel, per_pixel[0], atol=1e-15)
    assert truth['g_true'][1:].max() == 0.0


def test_least_squares_recovers_an_injected_ripple():
    """End to end on the operator itself: build a cube from a known ``g``,
    add foreground and noise, and solve the normal equations. The basis is
    exact here, so recovery is limited only by noise -- this is checking the
    plumbing, not the science."""
    rng = np.random.default_rng(5)
    nx, ny, nz = SHAPE
    evecs = legendre_basis(nz, N_MODES)

    cube, truth = groundspill_cube(FREQS, SHAPE, ripple_rms=1e-2,
                                   gradient=0.4, rng=rng)
    b = truth['basis']
    amps = rng.normal(scale=1.0, size=(nx * ny, N_MODES))
    foreground = (amps @ evecs).reshape(SHAPE)
    noise = rng.normal(scale=1e-3, size=SHAPE)
    data = (cube + foreground + noise).ravel()

    # Joint least squares over [f, g] only -- no signal block, no prior.
    design = np.concatenate([
        np.array([(np.eye(nx * ny)[p][:, None] * evecs[m]).ravel()
                  for p in range(nx * ny) for m in range(N_MODES)]),
        np.array([b.apply(np.eye(b.n_params)[i]) for i in range(b.n_params)]),
    ])
    sol = np.linalg.lstsq(design.T, data, rcond=None)[0]
    g_hat = sol[-b.n_params:].reshape(b.g_shape)

    err = np.abs(g_hat - truth['g_true']).max() / np.abs(truth['g_true']).max()
    assert err < 0.05, f'g recovered to {err:.3f}, expected <5%'


# ---------------------------------------------------------------------------
# Polarisation leakage
# ---------------------------------------------------------------------------

def test_galactic_rm_is_invisible_on_this_band():
    """The result that shapes the leakage model.

    The band spans lambda^2 = 0.0859-0.0953 m^2, so a Faraday depth of tens of
    rad/m^2 -- which is what the Galaxy actually supplies at these latitudes --
    turns through well under one cycle. It is smooth, the foreground absorbs
    all of it, and modelling it would add a null direction to Ug.
    """
    fg6 = legendre_basis(len(REAL_FREQS), 6)
    for rm in (10.0, 100.0, 300.0):
        tmpl = faraday_templates(REAL_FREQS, rm=rm)
        tmpl = tmpl / np.linalg.norm(tmpl, axis=1)[:, None]
        absorbed = np.mean(np.sum((fg6 @ tmpl.T) ** 2, axis=0))
        assert absorbed > 0.999, f'RM={rm}: {absorbed}'


def test_high_rm_leakage_survives():
    fg6 = legendre_basis(len(REAL_FREQS), 6)
    got = {}
    for rm in (500.0, 1000.0, 2000.0):
        tmpl = faraday_templates(REAL_FREQS, rm=rm)
        tmpl = tmpl / np.linalg.norm(tmpl, axis=1)[:, None]
        got[rm] = float(np.mean(np.sum((fg6 @ tmpl.T) ** 2, axis=0)))
    assert got[500.0] > 0.95            # still nearly all absorbed
    assert 0.1 < got[1000.0] < 0.3      # comparable to a 17.5 MHz ripple
    assert got[2000.0] < 0.1
    assert got[500.0] > got[1000.0] > got[2000.0]


def test_leakage_chirps_but_the_chirp_is_unresolved_here():
    """Periodic in lambda^2, so the local frequency period drifts as nu^3.

    The drift is real -- 17% across this band -- but it is NOT enough to
    spread the power in k. The k_parallel spacing is 2*pi/Lz = 0.0247, which
    at k ~ 0.074 is 33%, so a 17% drift falls inside one mode. An earlier
    version of this test's name claimed leakage was "not a single mode"; on a
    52 MHz band it effectively is. The chirp only resolves on a wider band.
    """
    l2 = lambda_squared(REAL_FREQS)
    # Local period in MHz at each end, from the phase gradient.
    grad = np.gradient(2 * RM_DEFAULT * l2, REAL_FREQS)
    period_lo, period_hi = 2 * np.pi / np.abs(grad[0]), 2 * np.pi / np.abs(grad[-1])
    ratio = period_hi / period_lo
    assert np.isclose(ratio, (REAL_FREQS[-1] / REAL_FREQS[0]) ** 3, rtol=1e-3)
    assert ratio > 1.15

    # ... and it is smaller than the k_parallel spacing, so it does not spread.
    k_spacing = 2 * np.pi / REAL_LZ
    k_ripple = ripple_wavenumber(17.5, REAL_FREQS[-1] - REAL_FREQS[0], REAL_LZ)
    assert (ratio - 1.0) < k_spacing / k_ripple


def test_leakage_basis_uses_2d_spatial_structure():
    """Leakage follows the beam, which has no reason to align with the scan."""
    b = leakage_basis(FREQS, SHAPE, rm=RM_DEFAULT, order=1)
    assert b.n_s == 3 and b.n_t == 2 and b.n_params == 6

    # Order-agnostic: at order 1 there must be exactly one constant term and
    # one gradient along each axis, whatever sequence poly2d_templates emits
    # them in.
    maps = [r.reshape(SHAPE[0], SHAPE[1]) for r in b.spatial]
    kinds = []
    for m in maps:
        varies_0 = np.ptp(m, axis=0).max() > 1e-12   # varies along axis 1
        varies_1 = np.ptp(m, axis=1).max() > 1e-12   # varies along axis 0
        kinds.append((varies_1, varies_0))
    assert sorted(kinds) == [(False, False), (False, True), (True, False)]


def test_poly2d_term_count():
    for order, n in ((0, 1), (1, 3), (2, 6)):
        assert poly2d_templates(SHAPE, order=order).shape[0] == n


# ---------------------------------------------------------------------------
# 1/f
# ---------------------------------------------------------------------------

def test_onef_covariance_is_symmetric_and_psd():
    cov = onef_covariance(48, alpha=1.0, knee_cycles=1.0)
    assert np.allclose(cov, cov.T, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_onef_power_grows_toward_large_scales():
    """It is 1/f: more variance on long scales than short ones. If this fails
    the sign of alpha is wrong somewhere and the process is blue, not red."""
    steep = onef_covariance(64, alpha=2.0, knee_cycles=4.0)
    vals = np.linalg.eigvalsh(steep)[::-1]
    assert vals[0] / vals[-1] > 50


def test_onef_basis_deflates_the_foreground():
    """The 1/f common mode is constant in frequency, so it is inside the
    foreground span to machine precision -- the same statement as for smooth
    ground spill. Deflating means the block carries only what is left."""
    fg = legendre_basis(SHAPE[2], N_MODES)
    basis, _ = onef_basis(FREQS, SHAPE, n_scan=2, n_spec=3, fg_basis=fg)
    leak = np.abs(fg @ basis.spectral.T).max()
    assert leak < 1e-8, f'spectral modes still overlap the foreground: {leak}'

    # Without deflation they do overlap, which is the point of the option.
    plain, _ = onef_basis(FREQS, SHAPE, n_scan=2, n_spec=3, fg_basis=None)
    assert np.abs(fg @ plain.spectral.T).max() > 1e-3


def test_onef_prior_variance_comes_from_the_eigenvalues():
    """Unlike ground spill, G here has a derivation rather than a guess: the
    KL eigenvalues ARE the prior variances."""
    fg = legendre_basis(SHAPE[2], N_MODES)
    basis, prior = onef_basis(FREQS, SHAPE, n_scan=3, n_spec=2, fg_basis=fg)
    assert prior.shape == basis.g_shape
    assert np.isclose(prior.sum(), 1.0)
    assert np.all(prior > 0)
    # Ordered: the leading mode carries the most variance in each direction.
    assert prior[0, 0] == prior.max()
    assert np.all(np.diff(prior[:, 0]) <= 1e-12)


def test_realise_and_recover_amplitudes():
    """A drawn realisation is exactly representable, so the least-squares
    amplitudes must return what was drawn."""
    fg = legendre_basis(SHAPE[2], N_MODES)
    basis, prior = onef_basis(FREQS, SHAPE, n_scan=2, n_spec=2, fg_basis=fg)
    cube, g_true = realise(basis, prior, rms=1e-3,
                           rng=np.random.default_rng(11))
    assert np.isclose(cube.std(), 1e-3, rtol=1e-10)
    assert np.allclose(best_fit_amplitudes(basis, cube), g_true, rtol=1e-8)


def test_best_fit_amplitudes_on_something_outside_the_basis():
    """A cube the basis cannot represent still gets its best projection, not
    an error -- that is what a 1/f recovery has to be scored against."""
    fg = legendre_basis(SHAPE[2], N_MODES)
    basis, _ = onef_basis(FREQS, SHAPE, n_scan=2, n_spec=2, fg_basis=fg)
    rng = np.random.default_rng(12)
    noise = rng.normal(size=SHAPE)
    g = best_fit_amplitudes(basis, noise)
    assert g.shape == basis.g_shape and np.all(np.isfinite(g))
    # The residual must be orthogonal to every basis vector: that is the
    # defining property of a least-squares fit.
    resid = noise.ravel() - basis.apply(g)
    assert np.abs(basis.adjoint(resid)).max() < 1e-8 * np.abs(g).max() + 1e-10
