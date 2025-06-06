from . import _painter
import sharedmem
import numpy
import multiprocessing
import sys

# Set multiprocessing method to fork on macOS to avoid pickling issues
if sys.platform == 'darwin' and multiprocessing.get_start_method(allow_none=True) != 'fork':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

def paint(pos, sml, data, shape, mask=None, np=0, periodic=False):
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

        periodic : bool, optional
          If True, apply periodic boundary conditions. Particles near edges
          will contribute to opposite sides of the image. Default is False.

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
    
    # Handle periodic boundary conditions
    if periodic:
        # Replicate particles that are near boundaries
        pos_list = [pos]
        sml_list = [sml]
        data_list = [[d] for d in data]
        mask_list = [mask] if mask is not None else [None]
        
        # Check which particles need replication
        width, height = shape[1], shape[0]
        
        # For each shift in the 3x3 grid (excluding center)
        for shift_x in [-1, 0, 1]:
            for shift_y in [-1, 0, 1]:
                if shift_x == 0 and shift_y == 0:
                    continue  # Skip the original position
                
                # Shift positions by full image dimensions
                dx = shift_x * width
                dy = shift_y * height
                
                # Create shifted positions for ALL particles
                shifted_pos = pos.copy()
                shifted_pos[:, 0] += dx
                shifted_pos[:, 1] += dy
                
                # Check which shifted particles could paint into the image
                # A particle can paint if its center +/- sml overlaps with [0, width] x [0, height]
                x_overlap = (shifted_pos[:, 0] + sml > 0) & (shifted_pos[:, 0] - sml < width)
                y_overlap = (shifted_pos[:, 1] + sml > 0) & (shifted_pos[:, 1] - sml < height)
                paint_mask = x_overlap & y_overlap
                
                if numpy.any(paint_mask):
                    pos_list.append(shifted_pos[paint_mask])
                    sml_list.append(sml[paint_mask])
                    
                    for i, d in enumerate(data):
                        data_list[i].append(d[paint_mask])
                    
                    if mask is not None:
                        mask_list.append(mask[paint_mask])
        
        # Concatenate all replicated particles
        pos = numpy.concatenate(pos_list)
        sml = numpy.concatenate(sml_list)
        data = [numpy.concatenate(d_list) for d_list in data_list]
        if mask is not None:
            mask = numpy.concatenate(mask_list)

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


