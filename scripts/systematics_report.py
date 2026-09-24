#!/usr/bin/env python
"""Report on a finished set of ``groundspill_injection.py`` arms.

Discovers every arm in an output directory, then writes convergence
diagnostics, the recovered power spectrum, the recovered H I map, and the
ground-spill amplitude posterior.

    python scripts/systematics_report.py --out outputs/onef_run

Two estimator choices here matter, and both follow ``docs/STATUS.md``.

**P(k) is measured on the footprint, per sample.** The ``Pk_trace`` files are
the box average over the whole cube, 41% of which is inpainting -- the sampler
filling flagged voxels from the prior with no data behind them. That sits up to
15x off at mid k. This recomputes P(k) from the saved signal cubes restricted
to the observed footprint and normalised by ``<w^2>``, which is the estimator
that agrees with the PCA transfer function.

It is computed **per sample and then averaged**, not from the posterior-mean
cube. The posterior mean is Wiener-suppressed, so its power is biased low --
that is an estimator difference, not a statement about the H I.

``Pk_trace`` is still used for the convergence diagnostics, where what matters
is the chain's own autocorrelation. Note that the ``--fix-S`` arm's
``Pk_trace`` is the raw binned power rather than an inverse-gamma draw, so its
trace is not like-for-like with the others; its footprint P(k) is.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgibbs import kbins_from_crop, load, load_l2021_cube, survey_grid  # noqa: E402

CROP = (slice(33, 103), slice(14, 59), slice(0, 250))
N_K_BINS = 5

# Categorical hues, assigned in fixed order and never cycled.
COLOURS = {'clean': '#2a78d6', 'off': '#eb6834', 'on': '#1baf7a',
           'onfixedS': '#eda100', 'cleanfixedS': '#e87ba4'}
LABELS = {'clean': 'control: no systematic injected',
          'off': 'contaminated, s + f',
          'on': 'contaminated, s + f + g',
          'onfixedS': 'contaminated, s + f + g, S held (diagnostic)',
          'cleanfixedS': 'control, S held (diagnostic)'}
ORDER = ['clean', 'cleanfixedS', 'off', 'on', 'onfixedS']

#: Arms that carry a posterior band. The two that are actually being compared
#: -- the control and the realistic configuration. The rest are plotted as bare
#: lines: they are there for visibility, and five overlapping bands on one axis
#: read as noise.
BAND_ARMS = ('clean', 'on')
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', default='outputs/groundspill_run1')
    p.add_argument('--burn', type=int, default=None,
                   help='override the burn-in recorded by each run')
    p.add_argument('--figdir', default=None,
                   help='where to write figures (default: <out>/figures)')
    p.add_argument('--channel', type=int, default=125,
                   help='frequency channel for the map panels')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover(out):
    """Every arm with a metadata file, in a fixed display order."""
    arms = {}
    for path in sorted(glob.glob(os.path.join(out, 'run_*_meta.json'))):
        meta = json.loads(Path(path).read_text())
        arms[meta['arm']] = meta
    if not arms:
        raise SystemExit(f'no run_*_meta.json under {out}')
    return {a: arms[a] for a in ORDER if a in arms}


def truth_cubes(meta):
    """Rebuild the H I cube and footprint mask the run was built on."""
    real = load_l2021_cube()[CROP]
    flag = (real != 0).astype(float)
    hi = load('Fastbox_cube_cropped.npy')[:, :, :real.shape[2]]
    grid = survey_grid(CROP, real.shape)
    return hi, flag, grid


def sample_cubes(out, arm, burn, thin):
    """Post-burn signal cubes, in sample order. Thinned on disk by the run."""
    pat = os.path.join(out, 'samples', f's_cube*_{arm}_.npy')
    got = []
    for path in glob.glob(pat):
        m = re.search(r's_cube(\d+)_', os.path.basename(path))
        if m and int(m.group(1)) >= burn:
            got.append((int(m.group(1)), path))
    return [p for _, p in sorted(got)]


def pk_traces(out, arm, burn, n_samples):
    return np.array([
        np.load(os.path.join(out, 'samples', f'Pk_trace{r}_{arm}_.npy'))
        for r in range(burn, n_samples)])


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def make_pk_footprint(idxs, occupied, flag):
    w2 = (flag ** 2).mean()

    def pk(cube):
        mu = cube[flag == 1].mean()
        A = np.fft.fftn((cube - mu) * flag, norm='ortho').flatten()
        p = (A * np.conj(A)).real
        return np.array([p[idxs == b].mean() for b in occupied]) / w2

    return pk


def tau_int(chain, c=5.0):
    """Integrated autocorrelation time, Sokal's automatic window.

    Returned in units of *stored* samples. ESS = len(chain) / tau.

    The variance test is deliberately relative. ``np.allclose(x, 0)`` would
    use an absolute tolerance of 1e-8 and so declare a perfectly healthy
    high-k chain dead -- P(k) in the top bin is ~6e-8 K^2 here.
    """
    x = np.asarray(chain, dtype=float)
    n = len(x)
    if n < 10 or np.ptp(x) == 0:
        return np.nan
    x = x - x.mean()
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    if acf[0] <= 0:
        return np.nan
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    window = np.arange(n) < c * taus
    idx = np.argmin(window) if not window.all() else n - 1
    return float(max(taus[idx], 1.0))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out = args.out
    figdir = args.figdir or os.path.join(out, 'figures')
    os.makedirs(figdir, exist_ok=True)

    arms = discover(out)
    first = next(iter(arms.values()))
    hi, flag, grid = truth_cubes(first)
    shape = hi.shape

    sig_k = np.array(first['sig_k'])
    _, idxs, _ = kbins_from_crop((hi + 1) * flag, grid.box_dims,
                                 max_bins=N_K_BINS)
    occupied = np.unique(idxs)
    occupied = occupied[occupied > 0]
    assert np.allclose(sig_k, first['sig_k'])

    pk_fp = make_pk_footprint(idxs, occupied, flag)
    true_pk = pk_fp(hi)
    hit = first['ripple_bin']

    print(f'arms         : {", ".join(arms)}')
    print(f'ripple       : k = {first["ripple_k"]:.4f} Mpc^-1 -> bin {hit}')
    print(f'injected     : ripple {first["ripple_rms"]:.2e} K, '
          f'H I rms {first["hi_rms"]:.2e} K\n')

    # ---- gather ------------------------------------------------------
    res = {}
    for arm, meta in arms.items():
        burn = args.burn if args.burn is not None else meta['burn']
        paths = sample_cubes(out, arm, burn, meta.get('thin', 1))
        if not paths:
            print(f'  {arm}: no signal cubes past burn-in {burn}, skipping')
            continue
        per_sample, mean_cube = [], np.zeros(shape)
        for path in paths:
            cube = np.load(path).astype(float)
            per_sample.append(pk_fp(cube))
            mean_cube += cube
        mean_cube /= len(paths)
        res[arm] = dict(
            meta=meta, burn=burn, n_cubes=len(paths),
            pk=np.array(per_sample), mean_cube=mean_cube,
            trace=pk_traces(out, arm, burn, meta['n_samples']))
        print(f'  {arm:9s}: {meta["n_samples"]} samples, burn {burn}, '
              f'{len(paths)} cubes for P(k)')

    # ---- convergence --------------------------------------------------
    print('\nCONVERGENCE (from Pk_trace, per stored sample)')
    print('  A tau_int estimate needs a chain ~50x longer than tau to be '
          'trustworthy;\n  anything flagged (!) below is a lower bound, not a '
          'measurement.')
    print('  arm        bin      tau_int      ESS')
    for arm, r in res.items():
        n = len(r['trace'])
        for i in range(len(sig_k)):
            t = tau_int(r['trace'][:, i])
            ess = n / t if t == t else np.nan
            flag_short = '  (!)' if t == t and n < 50 * t else ''
            print(f'  {arm:9s}  {i:^3d}   {t:9.1f}  {ess:7.0f}{flag_short}')

    # ---- power spectrum ----------------------------------------------
    print('\nFOOTPRINT-RESTRICTED P(k), mean over samples, ratio to true H I')
    header = '  bin       k     true P(k)  ' + ''.join(
        f'{a:>15s}' for a in res)
    print(header)
    for i, k in enumerate(sig_k):
        row = f'  {i:^3d} {k:7.4f}  {true_pk[i]:.3e}  '
        for arm, r in res.items():
            row += f'{r["pk"][:, i].mean() / true_pk[i]:15.2f}'
        print(row + ('   <-- ripple' if i == hit else ''))

    # ---- g recovery ---------------------------------------------------
    for arm, r in res.items():
        gp = os.path.join(out, 'samples', f'g_trace{r["burn"]}_{arm}_.npy')
        if not os.path.exists(gp):
            continue
        g = np.array([
            np.load(os.path.join(out, 'samples', f'g_trace{rr}_{arm}_.npy'))
            for rr in range(r['burn'], r['meta']['n_samples'])])
        truth_g = np.array(r['meta']['g_true'])
        r['g'], r['g_true'] = g, truth_g
        label = r['meta'].get('systematic', 'systematic').upper()
        print(f'\n{label} AMPLITUDES, arm "{arm}" (K)')
        print('  spatial spectral       true      recovered        of true')
        for i in range(truth_g.shape[0]):
            for j in range(truth_g.shape[1]):
                t, m, sd = truth_g[i, j], g[:, i, j].mean(), g[:, i, j].std()
                frac = f'{m / t:8.1%}' if abs(t) > 1e-12 else '       -'
                print(f'  {i:^7d} {j:^8d}  {t:+.3e}  {m:+.3e} +- {sd:.1e}'
                      f'  {frac}')

    figures(args, res, sig_k, true_pk, hit, hi, flag, figdir)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(args, res, sig_k, true_pk, hit, hi, flag, figdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 150,
        'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.linewidth': 0.8, 'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': GRID, 'grid.linewidth': 0.6,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'legend.frameon': False, 'font.size': 9, 'lines.linewidth': 1.8,
    })
    DIV = LinearSegmentedColormap.from_list(
        'gs_div', ['#0d366b', '#2a78d6', '#f0efec', '#e34948', '#8f2320'])

    def bare(ax):
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    # --- 1. recovered P(k), with the posterior spread ------------------
    # Three panels because one cannot do the job: the contaminated bin is 88x
    # and the interesting structure elsewhere is a few per cent, so a single
    # axis either hides the spike or hides the error bars.
    fig, (ax, axr, axz) = plt.subplots(1, 3, figsize=(15.5, 4.6),
                                       gridspec_kw={'width_ratios': [1.3, 1, 1]})
    ax.plot(sig_k, true_pk, color=INK, marker='*', ms=9, lw=2.0, zorder=6,
            label='true injected H I')
    for arm, r in res.items():
        lo, mid, up = np.percentile(r['pk'], [16, 50, 84], axis=0)
        r['lo'], r['mid'], r['up'] = lo, mid, up
        if arm in BAND_ARMS:
            ax.fill_between(sig_k, lo, up, color=COLOURS[arm], alpha=0.25, lw=0)
            axr.fill_between(sig_k, lo / true_pk, up / true_pk,
                             color=COLOURS[arm], alpha=0.25, lw=0)
        ax.plot(sig_k, mid, color=COLOURS[arm], marker='o', ms=5,
                label=LABELS[arm])
        axr.plot(sig_k, mid / true_pk, color=COLOURS[arm], marker='o', ms=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$k$ [Mpc$^{-1}$]'); ax.set_ylabel(r'$P(k)$ [K$^2$]')
    ax.set_title('Recovered power spectrum, footprint-restricted', loc='left')
    ax.legend(loc='lower left', fontsize=8)
    bare(ax)

    axr.axhline(1.0, color=INK, lw=1.2)
    axr.axvline(sig_k[hit], color=MUTED, lw=1.0, ls=(0, (2, 2)))
    axr.annotate('ripple bin', xy=(sig_k[hit], 1.0), xytext=(6, 8),
                 textcoords='offset points', fontsize=8.5, color=INK2)
    axr.set_xscale('log'); axr.set_yscale('log')
    axr.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    axr.set_ylabel('recovered / true')
    axr.set_title('Ratio to truth, all bins', loc='left')
    bare(axr)

    # Zoom: the contaminated bin dropped, a linear scale, and the posterior
    # bands finally large enough to read. They are 0.5-10% wide, so on the log
    # panels to the left they are thinner than the line.
    keep = [i for i in range(len(sig_k)) if i != hit]
    for arm, r in res.items():
        err = np.vstack([(r['mid'] - r['lo'])[keep] / true_pk[keep],
                         (r['up'] - r['mid'])[keep] / true_pk[keep]])
        if arm in BAND_ARMS:
            axz.errorbar(sig_k[keep], r['mid'][keep] / true_pk[keep], yerr=err,
                         color=COLOURS[arm], marker='o', ms=5, capsize=3,
                         lw=1.8, elinewidth=1.4)
        else:
            axz.plot(sig_k[keep], r['mid'][keep] / true_pk[keep],
                     color=COLOURS[arm], marker='o', ms=4, lw=1.2, alpha=0.75)
    axz.axhline(1.0, color=INK, lw=1.2)
    axz.set_xscale('log')
    axz.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    axz.set_ylabel('recovered / true')
    axz.set_title(f'Zoom: bin {hit} dropped, 16-84th percentile', loc='left')
    bare(axz)

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'power_spectrum.png'),
                bbox_inches='tight')

    # --- 2. the recovered H I map --------------------------------------
    chan = args.channel
    inside2d = flag[:, :, chan].astype(bool)

    def masked(field):
        """Blank the unobserved pixels, and remove the footprint mean.

        The sampler leaves the DC mode free on purpose (``S[dc] = 1e30``), so
        an overall offset carries no information and comparing raw means would
        just show that.
        """
        field = field - field[inside2d].mean()
        return np.ma.masked_where(~inside2d, field)

    short = {'clean': 'control (no systematic)', 'off': 's + f',
             'on': 's + f + g', 'onfixedS': 's + f + g, S held',
             'cleanfixedS': 'control, S held'}
    panels = [('true injected H I', masked(hi[:, :, chan]))]
    panels += [(short.get(a, a), masked(r['mean_cube'][:, :, chan]))
               for a, r in res.items()]

    # Scale on a robust percentile of the TRUTH, not its maximum: the H I is
    # lognormal, so one 14-sigma voxel otherwise washes out every panel. A
    # contaminated arm is then supposed to saturate -- that is the message.
    _t = hi[:, :, chan] - hi[:, :, chan][inside2d].mean()
    lim = float(np.percentile(np.abs(_t[inside2d]), 99))
    cmap = DIV.copy()
    cmap.set_bad('#f2f1ee')

    fig, axes = plt.subplots(1, len(panels), figsize=(2.75 * len(panels), 3.1))
    for ax, (name, field) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(field.T, origin='lower', cmap=cmap, aspect='auto',
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
        ax.set_title(name, fontsize=8.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), label='T [K]',
                 pad=0.012, shrink=0.85, extend='both')
    fig.suptitle(f'Posterior-mean signal map, channel {chan}  '
                 f'(shared scale, 99th percentile of the truth)',
                 x=0.02, ha='left', fontsize=10, color=INK)
    fig.savefig(os.path.join(figdir, 'hi_map.png'), bbox_inches='tight')

    # --- 3. how well does the map correlate with the truth? -------------
    inside = flag.astype(bool)
    fig, axes = plt.subplots(1, len(res), figsize=(3.3 * len(res), 3.2),
                             squeeze=False)
    for ax, (arm, r) in zip(axes[0], res.items()):
        # Mean-subtracted inside the footprint, for the same reason the map is.
        x = hi[inside] - hi[inside].mean()
        y = r['mean_cube'][inside] - r['mean_cube'][inside].mean()
        rho = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        # Clip to the bulk: the H I is lognormal, so a handful of voxels at
        # 180 mK otherwise squeeze everything else into one corner. The clip
        # has to reach hexbin as `extent` -- setting the axis limits afterwards
        # would leave the cells sized for the full range, i.e. four of them.
        clip = float(np.percentile(np.abs(x), 99.5)) * 1e3
        ax.hexbin(x * 1e3, y * 1e3, gridsize=45, bins='log', mincnt=1,
                  cmap='Blues', linewidths=0,
                  extent=(-clip, clip, -clip, clip))
        ax.set_xlim(-clip, clip); ax.set_ylim(-clip, clip)
        ax.plot([-clip, clip], [-clip, clip], color=MUTED, lw=1.0,
                ls=(0, (4, 3)), zorder=3)
        ax.set_title(f'{arm}   r = {rho:.3f}   slope = {slope:.2f}',
                     fontsize=9, color=COLOURS[arm])
        ax.set_xlabel('true H I [mK]')
        ax.grid(False)
        bare(ax)
        r['rho'], r['slope'] = rho, slope
    axes[0][0].set_ylabel('posterior mean [mK]')
    fig.suptitle('Recovered signal against the truth, footprint voxels only',
                 x=0.02, ha='left', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'hi_correlation.png'),
                bbox_inches='tight')

    # --- 4. traces: is the chain converged, and does g decay? -----------
    has_g = [a for a in res if 'g' in res[a]]
    fig, axes = plt.subplots(2 if has_g else 1, 1,
                             figsize=(7.6, 5.6 if has_g else 3.2), squeeze=False)
    ax = axes[0][0]
    for arm, r in res.items():
        n = np.arange(r['burn'], r['burn'] + len(r['trace']))
        ax.plot(n, r['trace'][:, hit], color=COLOURS[arm], lw=0.7, alpha=0.85,
                label=LABELS[arm])
    ax.axhline(true_pk[hit], color=INK, lw=1.4)
    ax.annotate('true H I power', xy=(ax.get_xlim()[1], true_pk[hit]),
                xytext=(-4, 5), textcoords='offset points', ha='right',
                fontsize=8.5, color=INK)
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(k)$ bin %d [K$^2$]' % hit)
    ax.set_title(f'Bin {hit} trace — the contaminated bin', loc='left')
    ax.legend(loc='upper right', fontsize=8)
    bare(ax)

    if has_g:
        axg = axes[1][0]
        for arm in has_g:
            r = res[arm]
            n = np.arange(r['burn'], r['burn'] + len(r['g']))
            axg.plot(n, r['g'][:, 0, 0], color=COLOURS[arm], lw=0.7,
                     alpha=0.85, label=LABELS[arm])
        axg.axhline(res[has_g[0]]['g_true'][0, 0], color=INK, lw=1.4)
        axg.annotate('true amplitude',
                     xy=(axg.get_xlim()[1], res[has_g[0]]['g_true'][0, 0]),
                     xytext=(-4, 5), textcoords='offset points', ha='right',
                     fontsize=8.5, color=INK)
        axg.set_xlabel('sample')
        axg.set_ylabel('g[0,0] [K]')
        axg.set_title('Ground-spill amplitude trace', loc='left')
        axg.legend(loc='lower right', fontsize=8)
        bare(axg)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'traces.png'), bbox_inches='tight')

    print(f'\nfigures -> {figdir}/'
          '{power_spectrum,hi_map,hi_correlation,traces}.png')


if __name__ == '__main__':
    main()
