"""
The Painter module for Gaepsi2.

This module provides functionality to paint SPH particles onto a 2D image
using SPH kernel interpolation. It is a wrapper around the optimized C/Cython
implementation in _painter.
"""

from typing import List, Tuple, Union, Optional, Any
import numpy as np
import sharedmem
from . import _painter


def paint(
        pos: np.ndarray,
        sml: np.ndarray,
        data: List[np.ndarray],
        shape: Tuple[int, int],
        mask: Optional[np.ndarray] = None,
        np: int = 0
) -> np.ndarray:
    """Use SPH kernel to splat particles to an image.

    Parameters
    ----------
    pos : array_like
        (..., >2) position of particles. Only two first
        columns are used. In device coordinate.

    data : list of array_like
        (Nc, ...) or (...). Weight to use for painting.
        Nc channels will be produced on the device.
        If the array is 1d, Nc = 1.

    sml : array_like
        Smoothing length (half of effective size).
        In device coordinate; only correct in isotropic
        cameras.

    shape : tuple
        (w[0], w[1]) the size of the device.
        Should enclose pos[..., 0] and pos[..., 1].

    mask : array_like, boolean, optional
        If provided, elements with False will not be painted.

    np : int, optional
        Number of processes for multiprocessing. 0 for single-processing.
        None for all available cores.

    Returns
    -------
    image: array_like
        (Nc, shape[0], shape[1])

    Notes
    -----
    Remember to transpose for imshow and pmesh to correctly put x horizontally.

    Examples
    --------
    >>> # Create simple test data
    >>> pos = np.array([[5, 5], [15, 15]])
    >>> sml = np.array([2.0, 3.0])
    >>> data = np.array([1.0, 2.0])
    >>> 
    >>> # Paint to a 20x20 image
    >>> image = paint(pos, sml, [data], (20, 20))
    >>> 
    >>> # Display with matplotlib
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(image[0], origin='lower')
    >>> plt.colorbar()
    >>> plt.show()
    """
    if len(np.shape(data)) == 1:
        data = [data]

    with sharedmem.MapReduce(np=np) as pool:
        if pool.np > 0:
            nbuf = pool.np
        else:
            nbuf = 1

        buf = sharedmem.empty([nbuf, len(data)] + list(shape), dtype='f4')
        buf[:] = 0
        chunksize = 1024 * 8

        def work(i):
            sl = slice(i, i + chunksize)
            datas = [d[sl] for d in data]
            if mask is not None:
                masks = mask[sl]
            else:
                masks = None
            _painter.paint(pos[sl], sml[sl], np.array(datas),
                           buf[pool.local.rank], masks)

        pool.map(work, range(0, len(pos), chunksize))

    return np.sum(buf, axis=0)