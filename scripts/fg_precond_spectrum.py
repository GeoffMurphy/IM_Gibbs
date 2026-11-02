#!/usr/bin/env python
"""Spectrum of the preconditioned foreground block on a sub-box of the real mask.

Reproduces the preconditioner table in docs/STATUS.md ("The beam in the
MODEL", finding 3). Forms the exact foreground block A_ff = F^-1 + N^-1 Uf^T
B W B Uf densely on a small sub-box of the live crop (with the measured beam
and the real mask), then for each candidate preconditioner reports LGMRES
and CG iteration counts and the eigenvalue range of M^-1 A_ff.

    python scripts/fg_precond_spectrum.py 10 38 5 25 [beam_table.npz]

x0 x1 y0 y1 select the sub-box. Without a beam table it uses a chromatic
Gaussian (1.5 deg at band centre, index 1); the table in STATUS.md was
measured with a tabulated measured beam, which is not distributed. The
sub-box is periodic, so the beam wraps at its edges; that does not matter for
comparing preconditioners. F is the value from a full-grid run.
"""
import sys, time
from pathlib import Path
import numpy as np, scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, lgmres, cg
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imgibbs import *
import imgibbs.grid as g

CROP = (slice(33, 103), slice(14, 59), slice(0, 250))
full = load_l2021_cube()[CROP]
sx, sy = slice(int(sys.argv[1]), int(sys.argv[2])), slice(int(sys.argv[3]), int(sys.argv[4]))
cube = full[sx, sy]
shape = cube.shape; nx, ny, nz = shape; P = nx * ny
freqs = g.NU0 + np.arange(g.CH0, g.CH1)[CROP[2]] * (g.NU1 - g.NU0) / g.NCHAN
w = (cube != 0).astype(float).ravel()
print(f'sub-box {shape}, voxel fill {w.mean():.2f}, empty pixels {np.mean(w.reshape(P, nz).sum(1) == 0):.2f}')
Ni = 1 / (16.0**2 / ((1712 - 856) / 4096 * 1e6 * 1000))
nm = 6
ev = np.linalg.qr(np.polynomial.legendre.legvander(np.linspace(-1, 1, nz), nm - 1))[0].T
F = np.array([1.757e3, 1.46e-1, 1.60e-4, 1.34e-5, 1.02e-5, 5.39e-6])   # from the full-grid run
S = np.full(P * nz, 1e-5)
beam = (load_beam_table(sys.argv[5], freqs, shape) if len(sys.argv) > 5
        else gaussian_beam(freqs, shape, 1.5, freqs.mean(), 1.0))
rs = (nx, ny, nz // 2 + 1); rl = int(np.prod(rs)); fs = (nx, ny, nm); fl = P * nm; n = 2 * rl + fl
sl = slice(2 * rl, n)

def A11_mv(v, b_=beam):
    x = np.zeros(n); x[sl] = v
    return construct_A(x, S, Ni * w, F, w, ev, rl, rs, fl, fs, shape, beam=b_)[sl]

t = time.time()
tag = Path(sys.argv[5]).stem if len(sys.argv) > 5 else 'gauss'
cache = Path('outputs') / f'A11_{tag}_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_{sys.argv[4]}.npy'
cache.parent.mkdir(exist_ok=True)
if cache.exists():
    A = np.load(cache)
else:
    A = np.array([A11_mv(e) for e in np.eye(fl)]).T
    A = 0.5 * (A + A.T)
    np.save(cache, A)
print(f'dense A11 {A.shape} in {time.time()-t:.0f} s; symmetric err {np.abs(A - A.T).max():.1e}')

def spectrum(name, Minv):
    """Eigenvalues of M^-1 A (similar to a symmetric matrix when M is SPD)."""
    lam = np.sort(np.real(np.linalg.eigvals(Minv @ A)))
    rng_ = np.random.default_rng(0); b = rng_.normal(size=fl)
    cnt = [0]
    def mv(v): cnt[0] += 1; return A @ v
    lgmres(LinearOperator((fl, fl), matvec=mv), b, rtol=1e-6, atol=0,
           M=LinearOperator((fl, fl), matvec=lambda v: Minv @ v), maxiter=2000)
    c2 = [0]
    def mv2(v): c2[0] += 1; return A @ v
    cg(LinearOperator((fl, fl), matvec=mv2), b, rtol=1e-6, atol=0,
       M=LinearOperator((fl, fl), matvec=lambda v: Minv @ v), maxiter=5000)
    q = np.quantile(lam, [0, 0.01, 0.5, 0.99, 1])
    print(f'{name:30s} lgmres {cnt[0]:5d} cg {c2[0]:5d} | eig min {q[0]:.2e} p1 {q[1]:.2e} med {q[2]:.2e} p99 {q[3]:.2e} max {q[4]:.2e} | kappa {q[4]/q[0]:.1e}', flush=True)

# P0: no preconditioner
spectrum('none', np.eye(fl))
# P1: current transverse-Fourier (uniform coverage)
P1 = construct_preconditioner(S, Ni, F, ev, rl, rs, fl, fs, shape, beam=beam)
M1 = np.array([P1(np.r_[np.zeros(2 * rl), e])[sl] for e in np.eye(fl)]).T
spectrum('Fourier (current)', M1)
# P2: exact mask, achromatic band-mean beam
kern = np.fft.irfftn(beam.transfer.mean(axis=2), s=(nx, ny), axes=(0, 1))
ix, iy = np.unravel_index(np.arange(P), (nx, ny))
Bm = kern[(ix[:, None] - ix[None, :]) % nx, (iy[:, None] - iy[None, :]) % ny]
Om = np.einsum('mz,nz,pz->pmn', ev, ev, w.reshape(P, nz))
K = np.zeros((P, nm, P, nm))
for a in range(nm):
    for b in range(nm):
        K[:, a, :, b] = (Bm * Om[:, a, b][None, :]) @ Bm
A2 = Ni * K.reshape(fl, fl); A2[np.diag_indices(fl)] += np.tile(1 / F, P)
print(f'achromatic approx vs exact: rel Frobenius err {np.linalg.norm(A2 - A)/np.linalg.norm(A):.2e}')
spectrum('exact mask, achromatic beam', np.linalg.inv(A2))
# P3: exact mask, per-channel-group beams (G groups)
for G in (5, 25):
    K = np.zeros((P, nm, P, nm))
    for grp in np.array_split(np.arange(nz), G):
        kg = np.fft.irfftn(beam.transfer[:, :, grp].mean(axis=2), s=(nx, ny), axes=(0, 1))
        Bg = kg[(ix[:, None] - ix[None, :]) % nx, (iy[:, None] - iy[None, :]) % ny]
        Og = np.einsum('mz,nz,pz->pmn', ev[:, grp], ev[:, grp], w.reshape(P, nz)[:, grp])
        for a in range(nm):
            for b in range(nm):
                K[:, a, :, b] += (Bg * Og[:, a, b][None, :]) @ Bg
    A3 = Ni * K.reshape(fl, fl); A3[np.diag_indices(fl)] += np.tile(1 / F, P)
    print(f'{G}-group approx vs exact: rel Frobenius err {np.linalg.norm(A3 - A)/np.linalg.norm(A):.2e}')
    spectrum(f'exact mask, {G} beam groups', np.linalg.inv(A3))

# Mode split and stale F
def Kgroups(G, modes):
    md = len(modes)
    K = np.zeros((P, md, P, md))
    for grp in np.array_split(np.arange(nz), G):
        kg = np.fft.irfftn(beam.transfer[:, :, grp].mean(axis=2), s=(nx, ny), axes=(0, 1))
        Bg = kg[(ix[:, None] - ix[None, :]) % nx, (iy[:, None] - iy[None, :]) % ny]
        e = ev[modes][:, grp]
        Og = np.einsum('mz,nz,pz->pmn', e, e, w.reshape(P, nz)[:, grp])
        for a in range(md):
            for b in range(md):
                K[:, a, :, b] += (Bg * Og[:, a, b][None, :]) @ Bg
    return Ni * K.reshape(P * md, P * md)
def dense_inv(G, modes, Fuse):
    D = Kgroups(G, modes); D[np.diag_indices_from(D)] += np.tile(1 / Fuse[modes], P)
    return np.linalg.inv(D)
def embed(blocks):
    """blocks: list of (mode list, inverse matrix over those modes)"""
    Minv = np.zeros((fl, fl))
    for modes, Mi in blocks:
        idx = (np.arange(P)[:, None] * nm + np.array(modes)[None, :]).ravel()
        Minv[np.ix_(idx, idx)] = Mi
    return Minv
# Fourier block restricted to modes >= md, as a dense matrix
def fourier_inv(modes, Fuse):
    e = ev[modes]; b2 = beam.full_transfer() ** 2
    M = np.diag(1 / Fuse[modes]) + Ni * np.einsum('mz,xyz,nz->xymn', e, b2, e)
    Mi = np.linalg.inv(M)                                  # (nx, ny, md, md)
    md = len(modes)
    out = np.zeros((P * md, P * md))
    for j in range(P * md):
        v = np.zeros((nx, ny, md)); v.reshape(-1)[j] = 1
        fk = np.fft.fft2(v, axes=(0, 1), norm='ortho')
        out[:, j] = np.fft.ifft2(np.einsum('xymn,xyn->xym', Mi, fk), axes=(0, 1), norm='ortho').real.ravel()
    return out
allm = list(range(nm))
for md in (2, 3, 4):
    lo, hi = allm[:md], allm[md:]
    spectrum(f'dense modes<{md} (5 grp) + Fourier', embed([(lo, dense_inv(5, lo, F)), (hi, fourier_inv(hi, F))]))
# stale F: preconditioner built with F, system has F' (inverse-Wishart-sized changes)
Mfull = dense_inv(5, allm, F)
rng = np.random.default_rng(3)
for frac in (0.05, 0.2, 1.0):
    Fp = F * np.exp(rng.normal(scale=frac, size=nm))
    A_save = A.copy()
    A[np.diag_indices(fl)] += np.tile(1 / Fp - 1 / F, P)
    spectrum(f'full dense, F stale by ~{frac:.0%}', Mfull)
    A[:] = A_save
