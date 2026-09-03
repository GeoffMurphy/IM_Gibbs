"""Radial k-binning and the binned power spectrum estimator.

Two binners live here.

:func:`kbins_from_crop` is the one to use. It follows the crop automatically:
it measures the survey footprint from the data, takes ``k_min`` from the
footprint rather than the box, drops the degenerate kz = 0 plane, and picks the
bin count from occupancy.

:func:`make_kbins` is the older box-based binner, kept because the PCA
transfer-function benchmark is calibrated against it and because it is the
simpler thing to reach for on a fully-filled simulated cube. On the real
footprint it produces bins that no amount of sampling can constrain -- see the
notes on :func:`kbins_from_crop`.

Both return the same contract -- ``(sig_k, idxs)`` with ``idxs`` in
``0..nbins`` and 0 meaning "excluded" -- so :func:`bin_it` and the S-rebuild
loop in the sampler work with either.
"""

from __future__ import annotations

import numpy as np

from .grid import PIX_DEG


def make_kbins(shape, nbins, box_dims=None):
    """Radial k-bin assignments for a general (Nx, Ny, Nz) grid.

    The box-based binner: ``k_min`` is the longest mode the *grid* supports.
    On a fully-filled cube that is the right answer. On the real footprint it
    is not -- see :func:`kbins_from_crop`.

    Parameters
    ----------
    shape : tuple (Nx, Ny, Nz)
        Need not be cubic.
    nbins : int
    box_dims : tuple (Lx, Ly, Lz), optional
        Physical box dimensions in any consistent unit. If None, uses
        dimensionless pixel-frequency units.

    Returns
    -------
    sig_k : (nbins,) array
        Geometric-mean k per bin (log-space centroid).
    idxs : (Nx*Ny*Nz,) int array
        Bin index per mode; 0 is reserved for the DC mode.
    """
    Nx, Ny, Nz = shape

    if box_dims is not None:
        Lx, Ly, Lz = box_dims
        kx = 2 * np.pi * np.fft.fftfreq(Nx) * Nx / Lx
        ky = 2 * np.pi * np.fft.fftfreq(Ny) * Ny / Ly
        kz = 2 * np.pi * np.fft.fftfreq(Nz) * Nz / Lz
    else:
        kx, ky, kz = (np.fft.fftfreq(n) for n in (Nx, Ny, Nz))

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    Kmag = np.sqrt(KX**2 + KY**2 + KZ**2).flatten()

    k_nz = Kmag[Kmag > 0]
    k_edges = np.logspace(np.log10(k_nz.min()), np.log10(k_nz.max()), nbins + 1)

    # np.digitize returns 0 for k < k_edges[0] (the DC mode), 1..nbins for
    # normal modes, and nbins+1 for corner modes beyond kmax. Clip the latter
    # into the last bin.
    idxs = np.clip(np.digitize(Kmag, k_edges), 0, nbins)
    sig_k = np.sqrt(k_edges[:-1] * k_edges[1:])   # geometric mean

    return sig_k, idxs


def footprint_kmin(data_cube, box_dims, pix_deg=PIX_DEG, axis='short'):
    """Smallest transverse wavenumber the survey footprint actually spans.

    The footprint is measured from the data itself (non-zero voxels), so it
    follows the crop the same way ``box_dims`` does. ``axis="short"`` uses the
    narrow dimension of the band, which is the honest limit -- a transverse
    mode has to fit across the band as well as along it. ``axis="long"`` is the
    permissive choice.

    Returns
    -------
    k_min : float
    info : dict
        The measured footprint extents, for the run's metadata.
    """
    Lx, Ly, Lz = box_dims
    Nx, Ny, Nz = data_cube.shape

    # Comoving angular diameter distance, recovered from box_dims so this needs
    # no cosmology of its own and cannot drift out of sync with it.
    D_M = Lx / np.deg2rad(Nx * pix_deg)

    mask = np.asarray(data_cube != 0).any(axis=2)
    ys, xs = np.nonzero(mask)
    pts = np.stack([ys, xs]).astype(float)
    pts -= pts.mean(axis=1, keepdims=True)
    evals = np.linalg.eigvalsh(np.cov(pts))          # ascending
    # 2-sigma principal extents, in degrees then Mpc.
    short_deg, long_deg = 2 * np.sqrt(evals) * pix_deg
    extent_deg = short_deg if axis == 'short' else long_deg
    extent_mpc = np.deg2rad(extent_deg) * D_M
    return 2.0 * np.pi / extent_mpc, dict(
        D_M_mpc=float(D_M), short_deg=float(short_deg), long_deg=float(long_deg),
        short_mpc=float(np.deg2rad(short_deg) * D_M),
        long_mpc=float(np.deg2rad(long_deg) * D_M),
        fill_frac=float(mask.mean()))


def kbins_from_crop(data_cube, box_dims, pix_deg=PIX_DEG, min_modes=20,
                    max_bins=12, drop_kz0=True, kmin_axis='short',
                    kmin_override=None, verbose=True):
    """Choose k-bins for whatever crop ``data_cube``/``box_dims`` describe.

    What this fixes over :func:`make_kbins`:

    1. **k_min comes from the survey footprint, not the box.** The cropped cube
       is a bounding box around a diagonal drift-scan band that fills ~59% of
       it. Fourier modes longer than the band itself are set by the zero
       padding, not by data. Using the box's ``k_nz.min()`` therefore creates
       bins no amount of sampling can constrain.

    2. **The kz = 0 plane is dropped.** A kz = 0 mode is exactly constant along
       frequency -- the smoothest possible spectrum, and precisely what the
       foreground component absorbs. Signal and foreground are degenerate
       there, so the sampler splits that power by prior and the chain wanders.
       It costs 0.4% of the modes to remove.

    3. **The bin count is chosen from occupancy**, not fixed by hand. Bins with
       a handful of modes give an inverse-gamma posterior so broad it is
       useless. ``max_bins`` is a *ceiling*: the returned count may be lower.

    4. **The lowest bin edge is pinned to k_min.** ``logspace(log10(x), ...)``
       does not round-trip exactly; on some grids ``edges[0]`` lands a few
       1e-18 above the smallest mode, which then fails ``digitize`` and is
       silently binned with DC.

    Measured on the (70, 45, 250) crop, this is the difference between bin 1
    being 100% kz = 0 with tau_int = 4 and ESS = 190, and having real
    information in it.

    Returns
    -------
    sig_k : (nbins,)
        Geometric bin centres, indexed by ``bin_idx - 1``.
    idxs : (Nvox,)
        Bin index per voxel; 0 = excluded (DC, kz=0, k < k_min).
    meta : dict
        Every choice made, for the run's metadata file. ``meta['dc_index']``
        locates the true DC mode within the excluded set -- the sampler needs
        it, see the note at the bottom of this file.
    """
    Nx, Ny, Nz = data_cube.shape
    Lx, Ly, Lz = box_dims

    kx = 2 * np.pi * np.fft.fftfreq(Nx) * Nx / Lx
    ky = 2 * np.pi * np.fft.fftfreq(Ny) * Ny / Ly
    kz = 2 * np.pi * np.fft.fftfreq(Nz) * Nz / Lz
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
    KPAR = np.abs(KZ).ravel()

    if kmin_override is not None:
        k_min, fp = float(kmin_override), {}
    else:
        k_min, fp = footprint_kmin(data_cube, box_dims, pix_deg, kmin_axis)

    keep = K > 0
    if drop_kz0:
        keep &= KPAR > 0
    keep &= K >= k_min
    if not keep.any():
        raise ValueError(f'no modes survive k_min={k_min:.4g}; '
                         'check box_dims or pass kmin_override')

    k_hi = K[keep].max()

    # Largest bin count whose *emptiest* bin still clears min_modes.
    nbins, counts = None, None
    for nb in range(max_bins, 1, -1):
        edges = np.logspace(np.log10(k_min), np.log10(k_hi), nb + 1)
        edges[0], edges[-1] = k_min, k_hi      # pin: logspace round-trips imprecisely
        c = np.array([np.count_nonzero((K[keep] >= edges[b])
                                       & (K[keep] < edges[b + 1]))
                      for b in range(nb)])
        c[-1] += np.count_nonzero(K[keep] == k_hi)  # closed top edge
        if c.min() >= min_modes:
            nbins, counts, kept_edges = nb, c, edges
            break
    if nbins is None:
        raise ValueError(f'even 2 bins cannot reach min_modes={min_modes}; '
                         'loosen min_modes or widen the k range')

    idxs = np.zeros(K.size, dtype=int)
    b = np.digitize(K[keep], kept_edges) - 1
    idxs[keep] = np.clip(b, 0, nbins - 1) + 1
    sig_k = np.sqrt(kept_edges[:-1] * kept_edges[1:])

    # The DC mode needs different handling from the other excluded modes -- see
    # the note at the bottom of this file.
    dc_index = int(np.argmin(K))

    meta = dict(shape=tuple(map(int, data_cube.shape)),
                box_dims=tuple(map(float, box_dims)),
                k_min=float(k_min), k_max=float(k_hi), n_k_bins=int(nbins),
                min_modes=int(min_modes), drop_kz0=bool(drop_kz0),
                kmin_axis=kmin_axis, modes_per_bin=counts.tolist(),
                n_excluded=int(np.count_nonzero(idxs == 0)),
                dc_index=dc_index, sig_k=sig_k.tolist(), **fp)

    if verbose:
        print(f'[kbins] {data_cube.shape} box '
              f'({Lx:.0f},{Ly:.0f},{Lz:.0f}) Mpc')
        if fp:
            print(f'[kbins] footprint {fp["long_deg"]:.1f} x {fp["short_deg"]:.1f} deg '
                  f'= {fp["long_mpc"]:.0f} x {fp["short_mpc"]:.0f} Mpc, '
                  f'fill {fp["fill_frac"] * 100:.1f}%')
        print(f'[kbins] k_min {k_min:.4f} ({kmin_axis} axis), k_max {k_hi:.3f}, '
              f'{nbins} bins, kz=0 {"dropped" if drop_kz0 else "kept"}')
        print(f'[kbins] modes/bin {counts.tolist()}')
        print(f'[kbins] excluded {meta["n_excluded"]} of {K.size} voxels '
              f'({100 * meta["n_excluded"] / K.size:.1f}%) -> idxs 0')
    return sig_k, idxs, meta


def bin_it(cube, sig_k, idxs):
    """Group a flattened Fourier cube by |k| bin.

    Indexes ``sig_k[bin_idx - 1]`` rather than counting along, so the k values
    stay correct even if a middle bin is empty and gets skipped.

    Returns
    -------
    binned_s, k_bins : lists of arrays, one entry per occupied bin
    """
    binned_s, k_bins = [], []
    unique_idxs = np.unique(idxs)
    unique_idxs = unique_idxs[unique_idxs > 0]     # skip bin 0 (excluded)
    flat = cube.flatten()
    for bin_idx in unique_idxs:
        sses = flat[idxs == bin_idx]
        binned_s.append(sses)
        k_bins.append(np.ones(len(sses)) * sig_k[bin_idx - 1])
    return binned_s, k_bins


def power_spectrum(cube, sig_k, idxs, box_dims, mask=None, cube2=None):
    """Binned auto- (or cross-) power spectrum in Mpc^3.

    One estimator for every P(k) in the notebooks, so the uncleaned, cleaned
    and simulated spectra are guaranteed to be formed the same way.

    Three things it gets right that an inline ``fftn`` does not:

    1. The mean is taken over VALID voxels only and the masked region is held
       at zero. ``cube.mean()`` averages over the ~41% of voxels that are
       exactly 0, so subtracting it stamps the survey footprint into the field.
       That dumps spurious power at low k, and it hits the uncleaned cube far
       harder than the cleaned one (~3 orders of magnitude in the second k
       bin) -- which fakes part of the gap between the two curves.
    2. It multiplies by the voxel volume, so the result is P(k) in Mpc^3.
       ``fftn(norm='ortho')`` alone returns mode variance, not P(k).
    3. Errors divide by sqrt(number of modes IN THAT BIN), not sqrt(number of
       bins), which would give flat fractional errors across all k when they
       should shrink by ~100x from the first bin to the last.

    Parameters
    ----------
    cube : (Nx, Ny, Nz) field
    sig_k : bin centres from :func:`make_kbins` or :func:`kbins_from_crop`
    idxs : per-mode bin assignment from the same call (0 = excluded, dropped)
    box_dims : (Lx, Ly, Lz) in Mpc
    mask : bool array of valid voxels, or None to use every voxel
    cube2 : second field for a cross-spectrum; None for an auto-spectrum

    Returns
    -------
    Pk, Pk_err, n_modes
    """
    cube = np.asarray(cube, dtype=float)
    if mask is None:
        mask = np.ones(cube.shape, dtype=bool)

    vox_vol = np.prod(box_dims) / np.prod(cube.shape)   # Mpc^3 per voxel

    def prep(c):
        c = np.asarray(c, dtype=float)
        assert c.shape == cube.shape, f'shape mismatch: {c.shape} vs {cube.shape}'
        out = np.zeros(c.shape)
        out[mask] = c[mask] - c[mask].mean()      # mean over valid voxels only
        return np.fft.fftn(out, norm='ortho').flatten()

    f1 = prep(cube)
    f2 = f1 if cube2 is None else prep(cube2)

    nbins = len(sig_k)
    Pk = np.full(nbins, np.nan)
    n_modes = np.zeros(nbins, dtype=int)
    for b in range(1, nbins + 1):
        sel = (idxs == b)
        n_modes[b - 1] = sel.sum()
        if n_modes[b - 1]:
            Pk[b - 1] = np.mean((f1[sel] * np.conj(f2[sel])).real) * vox_vol

    with np.errstate(divide='ignore', invalid='ignore'):
        Pk_err = Pk / np.sqrt(n_modes)

    return Pk, Pk_err, n_modes


# ---------------------------------------------------------------------------
# How the excluded modes are handled in the sampler
# ---------------------------------------------------------------------------
#
# The S-rebuild in the Gibbs loop used to end with
#
#     S[idxs == 0] = 1e30    # DC mode: large variance = uninformative prior
#
# which was written when bin 0 held exactly ONE voxel, the DC mode. With kz = 0
# and the sub-footprint modes now also mapped to 0, it covers ~3200 voxels on
# the (70, 45, 250) crop -- and 1e30 gives that whole plane an essentially flat
# prior, which is the opposite of what we want. Those are precisely the modes
# degenerate with the foreground, so a flat prior invites the signal component
# to absorb foreground power there and destabilise the rest of the fit.
#
# Both the generation notebook and the sampler now do this instead:
#
#     S[idxs == 0] = 1e-12 * np.median(PkSample)   # suppress the degenerate plane
#     S[meta['dc_index']] = 1e30                   # keep the overall mean free
#
# This is a modelling choice, not a bug fix, and it has a consequence worth
# knowing: forcing s = 0 on the kz = 0 plane means the foreground component
# must account for all frequency-constant structure. That is correct if the
# foreground model has enough freedom; if it does not, the residual is pushed
# into the noise term instead. Worth checking the foreground residual if you
# change the number of foreground modes.
