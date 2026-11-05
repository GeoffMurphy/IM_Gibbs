"""The Gibbs chain: one function shared by the CLI and the notebooks.

Each iteration draws, in turn,

1. ``(s, f [, g]) | S, F, d`` -- one Gaussian constrained realisation, by
   solving ``A x = b`` with LGMRES (:func:`construct_A`, :func:`construct_b`);
2. ``g | s, f, d`` -- only when the systematic block is integrated out of
   step 1 (``gs_solve='marginal'``);
3. ``S | s`` -- inverse-gamma, one value per k-bin;
4. ``F | f`` -- inverse-Wishart over the foreground amplitudes.

``scripts/run_gibbs.py`` is a command-line wrapper round :func:`run_chain`;
``notebooks/1_end_to_end.ipynb`` calls it directly. The random draws are made
from NumPy's global generator in a fixed order, so a seed reproduces a chain
bit for bit.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres
from tqdm.auto import tqdm

from .covariance import (
    foreground_covariance_sampler as FCS,
    signal_covariance_sampler as SCS,
)
from .kbins import bin_it
from .linear_system import (
    MarginalNoise, MaskedForegroundBlock, Uf, Us, construct_A, construct_b,
    construct_preconditioner, rfft_adjoint_weight,
)


def legendre_basis(n_freq, n_modes):
    """The foreground basis: orthonormalised Legendre polynomials in frequency.

    Returns ``(n_modes, n_freq)`` with orthonormal rows.
    """
    poly = np.polynomial.legendre.legvander(np.linspace(-1, 1, n_freq),
                                            n_modes - 1).T
    return np.linalg.qr(poly.T)[0].T


def signal_prior(pk, idxs, dc_index):
    """Broadcast per-bin power ``pk`` onto the voxels: the diagonal ``S``.

    ``idxs == 0`` is DC plus the kz = 0 plane plus the sub-footprint modes. A
    flat 1e30 prior there would let the signal absorb foreground power in
    exactly the degenerate modes, so they are suppressed and only the true DC
    mode is left free to carry the overall mean.
    """
    pk = np.asarray(pk, dtype=float)
    occupied = np.unique(idxs)
    occupied = occupied[occupied > 0]
    S = np.zeros(len(idxs))
    for gg, bin_idx in enumerate(occupied):
        S[idxs == bin_idx] = pk[gg]
    S[idxs == 0] = 1e-12 * np.median(pk)
    S[dc_index] = 1e30
    return S


def resolve_gs_solve(gs_solve, beam):
    """``'auto'`` integrates the systematic out under a beam, else joint."""
    if gs_solve != 'auto':
        return gs_solve
    return 'marginal' if beam is not None else 'joint'


@dataclass
class Chain:
    """What :func:`run_chain` returns. Per-sample arrays have length
    ``n_samples``; ``signal_mean`` and ``fg_mean`` average samples
    ``mean_from`` onward."""

    pk: np.ndarray                 # (n_samples, n_bins), the S draws
    F: np.ndarray                  # (n_samples, n_modes), diag of F
    g: np.ndarray | None           # (n_samples, *g_shape)
    matvecs: np.ndarray            # per solve
    fg_refreshes: list
    signal_mean: np.ndarray        # real-space posterior-mean signal, K
    fg_mean: np.ndarray            # real-space posterior-mean foreground, K
    mean_from: int
    x: np.ndarray                  # the last sample
    S: np.ndarray                  # the last S
    elapsed: float
    f_prior_mean: str
    gs_solve: str | None
    visible_modes: list | None     # per bin, with the collapsed S step


def run_chain(data_cube, w, evecs, S, sig_k, idxs, dc_index, N, n_samples,
              *, tol=1e-6, seed=None,
              sys_basis=None, G=None, gs_solve='auto',
              beam=None, collapse=True, beam_threshold=1e-3,
              fg_dense_modes=3, fg_beam_groups=5, fg_refresh=20,
              f_prior_mean='auto', fix_S=False, mean_from=0,
              sample_dir=None, suffix='', save_x_every=1, save_S=False,
              progress=True, log=print):
    """Run one Gibbs chain.

    Parameters
    ----------
    data_cube : array (Nx, Ny, Nz)
        The data, K, with flagged voxels set to zero.
    w : array (Nx * Ny * Nz,)
        Weights; 0/1 for a footprint mask.
    evecs : array (n_modes, Nz)
        Foreground basis, orthonormal rows (:func:`legendre_basis`).
    S : array (Nx * Ny * Nz,)
        Starting signal covariance (:func:`signal_prior`).
    sig_k, idxs, dc_index :
        The k-binning, from :func:`imgibbs.kbins_from_crop`.
    N : float
        White noise variance per voxel, K^2.
    n_samples : int
    tol : float
        LGMRES relative tolerance. A sampling parameter, not just a speed one.
    seed : int, optional
        Seeds NumPy's global generator before the first draw.
    sys_basis, G :
        A systematic block (:class:`imgibbs.SystematicBasis`) and its prior
        variances, flattened. ``None`` for the three-block sampler.
    gs_solve : {'auto', 'joint', 'marginal'}
        Sample ``g`` inside the solve, or integrate it out and draw it after.
        ``'auto'`` is marginal with a beam (docs/STATUS.md, finding 9).
    beam : :class:`imgibbs.Beam`, optional
        Acts on signal and foreground, not the systematic.
    collapse : bool
        With a beam, draw ``S`` from the beam-visible modes only.
    beam_threshold : float
        Band-averaged ``|B|^2`` below which a mode counts as invisible.
    fg_dense_modes, fg_beam_groups, fg_refresh :
        The exact-mask foreground preconditioner used with a beam.
    f_prior_mean : {'auto', 'data', 'zero'}
        Centre of the foreground prior. ``'auto'`` is the data projection
        without a beam, zero with one (docs/STATUS.md).
    fix_S : bool
        Diagnostic: hold ``S`` and record the power it would have drawn.
    mean_from : int
        First sample in the posterior-mean maps.
    sample_dir : str, optional
        Write per-sample traces here, named as ``scripts/run_gibbs.py`` always
        has (``Pk_trace{i}{suffix}.npy`` and so on).
    save_x_every : int
        Write the full state ``x`` every Nth sample (0: never). 6 MB each.
    progress : bool
        Show a progress bar.
    log : callable
        Where setup messages go.

    Returns
    -------
    Chain
    """
    if seed is not None:
        np.random.seed(seed)

    shape = data_cube.shape
    n_freq = shape[2]
    n_modes = evecs.shape[0]
    occupied_bins = np.unique(idxs)
    occupied_bins = occupied_bins[occupied_bins > 0]
    gs_solve = None if sys_basis is None else resolve_gs_solve(gs_solve, beam)
    if sample_dir is not None:
        os.makedirs(sample_dir, exist_ok=True)

    # Collapsed S step. Modes the beam has removed carry (almost) no
    # information about S; drawing S from them as well lets S re-estimate
    # itself from its own prior draws -- a near random walk. Instead draw S
    # from the visible modes only (marginalising the rest), then apply it to
    # every mode in the bin. Exact as the excluded modes' information -> 0.
    kept = None
    if beam is not None and collapse:
        visible = beam.observable_modes(beam_threshold)
        idxs_S = np.where(visible, idxs, 0)
        kept = [int(np.sum(idxs_S == b)) for b in occupied_bins]
        total = [int(np.sum(idxs == b)) for b in occupied_bins]
        if min(kept) < 3:
            raise ValueError(f'beam_threshold {beam_threshold:g} leaves bins '
                             f'with {kept} visible modes')
        log('S step       : drawn from beam-visible modes only, '
            + ', '.join(f'{k}/{t}' for k, t in zip(kept, total)))
    else:
        idxs_S = idxs

    N_inv = 1 / N
    N_inv_scalar = 1.0 / N
    Nw_inv = (1 / N) * w

    # Shapes for packing/unpacking x = [Re(s), Im(s), f (, g)]
    rfft_shape = (shape[0], shape[1], shape[2] // 2 + 1)
    rfft_len = int(np.prod(rfft_shape))
    f_shape = (shape[0], shape[1], n_modes)
    f_len = int(np.prod(f_shape))
    # With g integrated out, the solve is over (s, f) alone and the noise
    # weights become an operator; g is drawn after each solve.
    marg = (MarginalNoise(Nw_inv, sys_basis, G) if gs_solve == 'marginal'
            else None)
    solve_basis = None if marg is not None else sys_basis
    A_noise = marg if marg is not None else Nw_inv
    g_len = 0 if solve_basis is None else solve_basis.n_params
    total_len = 2 * rfft_len + f_len + g_len
    g_mean = None if sys_basis is None else np.zeros(sys_basis.g_shape)

    # Starting points. The data's own projection onto the foreground basis
    # is the starting f and (f_prior_mean='data') the centre of its prior.
    s_mean = np.zeros(rfft_shape)                # zero-mean signal prior
    d_2d = data_cube.reshape(-1, n_freq)
    f_proj = (d_2d @ evecs.T).reshape(f_shape)
    f_prior = (f_prior_mean if f_prior_mean != 'auto'
               else ('zero' if beam is not None else 'data'))
    f_mean = f_proj if f_prior == 'data' else np.zeros_like(f_proj)
    log(f'fg prior     : centred on '
        f'{"the data projection" if f_prior == "data" else "zero"}')

    # Perturb before the inverse-Wishart draw: at f == f_proj exactly the
    # scatter matrix is singular, so FCS needs the spread.
    f_init = f_proj * np.random.normal(1.0, 0.05, f_proj.shape)
    F = np.diag(FCS(f_init.reshape(-1, n_modes)))

    x = np.concatenate([s_mean.real.flatten(), s_mean.imag.flatten(),
                        f_proj.flatten()]
                       + ([g_mean.flatten()] if solve_basis is not None
                          else []))

    A_calls = [0]
    matvecs = []

    def A_flat(vec):
        A_calls[0] += 1
        return construct_A(vec, S, A_noise, F, w, evecs, rfft_len, rfft_shape,
                           f_len, f_shape, shape,
                           sys_basis=solve_basis, G=G, beam=beam).flatten()

    L = LinearOperator(matvec=A_flat, rmatvec=A_flat, shape=(len(x), len(x)))

    # With a beam and a mask the default foreground preconditioner does not
    # converge (docs/STATUS.md); treat the stiff low-order modes exactly.
    fg_block = None
    if beam is not None and fg_dense_modes > 0:
        t0 = time.time()
        fg_block = MaskedForegroundBlock(beam, w, evecs, N_inv_scalar,
                                         n_dense=fg_dense_modes,
                                         n_groups=fg_beam_groups)
        fg_block.factor(F)
        log(f'fg precond   : exact-mask block, {fg_block.n_dense} modes, '
            f'{fg_beam_groups} beam groups, built in '
            f'{time.time() - t0:.0f} s, refreshed every {fg_refresh}')

    def preconditioner():
        apply = construct_preconditioner(S, N_inv_scalar, F, evecs, rfft_len,
                                         rfft_shape, f_len, f_shape, shape,
                                         sys_basis=solve_basis, G=G,
                                         beam=beam, fg_block=fg_block)
        if marg is not None:
            apply = marg.preconditioner(apply, C_marg, w_adj)
        return apply

    C_marg = w_adj = None
    if marg is not None:
        # The marginal system is A0 minus a rank-n_params term that the
        # block preconditioner does not see; correct it by Woodbury. The
        # coupling rows A_x^T W Ug e_i do not depend on S or F.
        zero_s, zero_f = np.zeros(rfft_shape), np.zeros(f_shape)
        C_marg = np.array([
            construct_b(S, N_inv, F, w, zero_s, zero_f, evecs, data_cube,
                        np.zeros(rfft_len), zero_f, np.zeros(w.size), shape,
                        beam=beam, weighted_data=row)
            for row in marg.WU])
        w_adj = rfft_adjoint_weight(rfft_shape, f_len)
    precond_apply = preconditioner()

    pk_trace, F_trace, g_trace = [], [], []
    signal_sum = np.zeros(shape)
    fg_sum = np.zeros(shape)
    n_mean = 0
    warned_q = False
    fg_baseline, since_refresh, fg_refreshes = None, 0, []

    def save(name, rr, arr):
        if sample_dir is not None:
            np.save(os.path.join(sample_dir, f'{name}{rr}{suffix}.npy'), arr)

    start = time.time()
    for rr in tqdm(range(n_samples), disable=not progress):
        x0 = x      # warm-start the solver from the previous sample
        if save_x_every and rr % save_x_every == 0:
            save('x_sample_', rr, x)

        # Omega draws: these turn the solve from a MAP estimate into a
        # posterior sample. Zero them for the MAP/Wiener-filter solution.
        ws = np.fft.rfftn(np.random.normal(size=shape), norm='ortho').flatten()
        wf = np.random.normal(size=f_shape)
        wd = np.random.normal(size=shape).flatten()
        wg = (None if sys_basis is None
              else np.random.normal(size=sys_basis.g_shape))

        b = construct_b(S, N_inv, F, w, s_mean, f_mean, evecs, data_cube,
                        ws, wf, wd, shape, sys_basis=solve_basis, G=G,
                        wg=wg, g_mean=g_mean, beam=beam,
                        weighted_data=(None if marg is None else
                                       marg.weighted_data(data_cube, wd, wg)))

        M_inv = LinearOperator((total_len, total_len), matvec=precond_apply)
        n_mv = A_calls[0]
        x, exit_code = lgmres(L, b.flatten(), x0=x0, rtol=tol, atol=0,
                              M=M_inv)
        matvecs.append(A_calls[0] - n_mv)
        if exit_code != 0:
            log(f'  [{rr}] lgmres did not converge (exit {exit_code})')

        # --- P(k) step: inverse-gamma draw per k-bin ---
        s = (x[0:rfft_len].reshape(rfft_shape)
             + x[rfft_len:2 * rfft_len].reshape(rfft_shape) * 1j)
        s = Us(s, False)
        f_ms = x[2 * rfft_len:2 * rfft_len + f_len].reshape(f_shape).real
        if rr >= mean_from:
            signal_sum += np.real(s).reshape(shape)
            fg_sum += Uf(evecs, f_ms, False).reshape(shape)
            n_mean += 1
        if marg is not None:
            # g | s, f, d: the residual after the (beamed) sky model.
            sky = s.flatten() + Uf(evecs, f_ms, False)
            if beam is not None:
                sky = beam.apply(sky)
            g_draw = marg.draw(data_cube.flatten() - sky,
                               np.random.normal(size=sys_basis.n_params))
        s = np.fft.fftn(s - np.mean(s), norm='ortho')

        binned_s, k_bins = bin_it(s, sig_k, idxs_S)
        if fix_S:
            # S is NOT updated; record the power the S draw would have used.
            PkSample = np.array([np.mean(np.abs(b) ** 2) for b in binned_s])
        else:
            _, PkSample = SCS(np.concatenate(binned_s),
                              np.concatenate(k_bins))
            S = signal_prior(PkSample, idxs, dc_index)

        if beam is not None and collapse and not fix_S:
            # The excluded modes have |B|^2 < threshold, so each carries at
            # most this data/prior precision ratio. If S has grown enough to
            # make that non-negligible, the collapse is no longer exact.
            q_excl = w.mean() * N_inv * beam_threshold * float(np.max(PkSample))
            if q_excl > 0.05 and not warned_q:
                log(f'  [{rr}] WARNING: excluded modes may now carry up to '
                    f'{q_excl:.2f} of their prior precision; lower '
                    f'beam_threshold')
                warned_q = True

        pk_trace.append(np.asarray(PkSample, dtype=float))
        save('Pk_trace', rr, PkSample)
        if save_S:
            save('S_trace', rr, S)

        # --- g: no covariance step. G is a prior, not a sampled quantity --
        # a single vector of n_params numbers does not identify its own
        # covariance the way Npix foreground amplitude vectors identify F.
        if sys_basis is not None:
            g_off = 2 * rfft_len + f_len
            g_now = (g_draw if marg is not None else
                     x[g_off:g_off + g_len].reshape(sys_basis.g_shape))
            g_trace.append(np.array(g_now))
            save('g_trace', rr, g_now)

        # --- F step: inverse-Wishart draw ---
        f_centered = (f_ms - f_mean).reshape((shape[0] * shape[1], n_modes))
        F = np.diag(FCS(f_centered))
        F_trace.append(np.diag(F).copy())
        save('F_trace', rr, F)

        if fg_block is not None:
            if fg_baseline is None:
                fg_baseline = matvecs[-1]
            since_refresh += 1
            if since_refresh >= fg_refresh or matvecs[-1] > 2 * fg_baseline:
                fg_block.factor(F)
                fg_refreshes.append(rr)
                fg_baseline, since_refresh = None, 0
        precond_apply = preconditioner()
        gc.collect()

    elapsed = time.time() - start
    return Chain(pk=np.array(pk_trace), F=np.array(F_trace),
                 g=np.array(g_trace) if g_trace else None,
                 matvecs=np.array(matvecs), fg_refreshes=fg_refreshes,
                 signal_mean=signal_sum / max(n_mean, 1),
                 fg_mean=fg_sum / max(n_mean, 1), mean_from=mean_from,
                 x=x, S=S, elapsed=elapsed, f_prior_mean=f_prior,
                 gs_solve=gs_solve, visible_modes=kept)


def load_chain(run_dir, suffix):
    """Read back a run written by ``scripts/run_gibbs.py`` as a :class:`Chain`.

    The per-sample traces come from ``<run_dir>/samples``; the posterior-mean
    maps are the ones the run saved, averaged from its ``burn`` onward.
    ``x`` and ``S`` are not kept by a run and come back as ``None``.
    """
    import json
    from pathlib import Path
    run = Path(run_dir)
    cfg = json.loads((run / f'run{suffix}config.json').read_text())
    n = cfg['n_samples']

    def trace(name):
        files = [run / 'samples' / f'{name}{i}{suffix}.npy' for i in range(n)]
        if not all(f.exists() for f in files):
            return None
        return np.array([np.load(f) for f in files])

    F = trace('F_trace')
    return Chain(pk=trace('Pk_trace'),
                 F=None if F is None else np.array([np.diag(f) for f in F]),
                 g=trace('g_trace'),
                 matvecs=np.load(run / f'matvecs{suffix}.npy'),
                 fg_refreshes=[], signal_mean=np.load(
                     run / f'signal_mean{suffix}.npy'),
                 fg_mean=np.load(run / f'fg_mean{suffix}.npy'),
                 mean_from=cfg['burn'], x=None, S=None, elapsed=float('nan'),
                 f_prior_mean=cfg['f_prior_mean'], gs_solve=cfg['gs_solve'],
                 visible_modes=cfg.get('beam_visible_modes_per_bin'))
