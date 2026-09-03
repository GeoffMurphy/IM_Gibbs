"""Locating the input cubes.

The simulated and derived products ship in ``data/``. The MeerKLASS L2021
observational cubes do NOT -- they are collaboration data and are not
redistributed here. Point :data:`DATA_DIR` at wherever you keep them, either by
setting ``IMGIBBS_DATA`` or by dropping them into ``data/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: Where the cubes live. Override with the ``IMGIBBS_DATA`` environment
#: variable; otherwise ``data/`` beside the repository root.
DATA_DIR = Path(os.environ.get(
    'IMGIBBS_DATA', Path(__file__).resolve().parent.parent / 'data'))

L2021_CUBE = 'L2021_polished_cube.npy'

_MISSING_L2021 = """\
Could not find {name} in {dir}.

The MeerKLASS L2021 cubes are not redistributed in this repository. To run the
notebooks on the real data, either

  * set IMGIBBS_DATA to the directory holding {name}, or
  * copy {name} into {dir}

It is built from the MeerKLASS 2021 map
    Nscan961_Tsky_cube_p0.3d_sigma4.0_iter2.fits
(0.3 deg pixels), keeping FITS channels 550-1050 -- a (133, 73, 500) cube in K.
See docs/STATUS.md, "Data provenance".
"""


def data_path(name: str) -> Path:
    """Absolute path to a file in the data directory. Does not check it exists."""
    return DATA_DIR / name


def load(name: str) -> np.ndarray:
    """Load a ``.npy`` from the data directory."""
    path = data_path(name)
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Set IMGIBBS_DATA or place '
                                f'the file in {DATA_DIR}.')
    return np.load(path)


def load_l2021_cube() -> np.ndarray:
    """Load the full (133, 73, 500) MeerKLASS L2021 cube, in K.

    Raises a message explaining where to get it if it is not present.
    """
    path = data_path(L2021_CUBE)
    if not path.exists():
        raise FileNotFoundError(
            _MISSING_L2021.format(name=L2021_CUBE, dir=DATA_DIR))
    return np.load(path)
