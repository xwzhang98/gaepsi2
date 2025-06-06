from . import _painter
import sharedmem
import numpy

def paint(pos, sml, data, shape, mask=None, np=0):
    """ Use SPH kernel to splat particles to an image.

        Parameters
        ----------
        pos : array_like
          (..., >=2) position of particles. Only the first two
          columns are used. In device coordinate

        sml : array_like
          smoothing length (half of effective size).
          In device coordinate; only correct for orthographic
          projections

        data : array_like
          (Nc, ...) or (...). Weight to use for painting.
          Nc channels will be produced on the device.
          If the array is 1d, Nc = 1
        
        shape : list, tuple
          (height, width) the size of the output image.
          Should enclose pos[..., 0] and pos[..., 1]

        mask : array_like, boolean, optional
          If provided, elements with False will not be painted.

        np : int, optional
          Number of processes for multiprocessing. 0 for single-processing.
          None for all available cores.

        Returns
        -------
        image: list of array_like
           List of (height, width) arrays, one for each data channel

        Notes
        -----
        Remember to transpose for imshow and pcolormesh to correctly put x horizontally.
    """
    # Input validation
    pos = numpy.asarray(pos)
    sml = numpy.asarray(sml)
    
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError("pos must be a 2D array with at least 2 columns")
    
    if len(sml) != len(pos):
        raise ValueError("sml must have same length as pos")
    
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("shape must be a 2-tuple (height, width)")
    
    if not all(isinstance(s, int) and s > 0 for s in shape):
        raise ValueError("shape must contain positive integers")
    
    if mask is not None:
        mask = numpy.asarray(mask, dtype=bool)
        if len(mask) != len(pos):
            raise ValueError("mask must have same length as pos")

    if len(numpy.shape(data)) == 1:
        data = [data]
    
    # Validate data arrays
    for i, d in enumerate(data):
        d = numpy.asarray(d)
        if len(d) != len(pos):
            raise ValueError(f"data[{i}] must have same length as pos")

    with sharedmem.MapReduce(np=np) as pool:
        if pool.np > 0: nbuf = pool.np
        else: nbuf = 1
        buf = sharedmem.empty([nbuf, len(data)] + list(shape), dtype='f4')
        buf[:] = 0
        chunksize = 1024 * 8

        def work(i):
            sl = slice(i, i + chunksize)
            datas = [d[sl] for d in data]
            if mask is not None: masks = mask[sl]
            else: masks = None
            _painter.paint(pos[sl], sml[sl], numpy.array(datas),
                    buf[pool.local.rank], masks)

        pool.map(work, range(0, len(pos), chunksize))
    return numpy.sum(buf, axis=0)


