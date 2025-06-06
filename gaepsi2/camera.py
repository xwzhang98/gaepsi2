"""
   Camera support in Gaepsi

   A camera is represented by a 4x4 transformation matrix like OpenGL.

   If it is done correctly, we use
   
   pos[:, :4] dot  matrix for the transformation.

   lookat: build the model/view matrix
   ortho:  the orthogonal perspective matrix

   the full transformation is ortho dot lookat

   apply: apply the transformation to a 3 d position vector,

   the device coordinate is [-1, 1]. 

   data_to_device transforms pos and sml together.

"""

import numpy
import sharedmem

# Global variables for multiprocessing workers
_worker_pos = None
_worker_matrix = None
_worker_shmout = None
_worker_xc = None
_worker_out = None
_worker_extent_params = None

class projectionmatrix(numpy.ndarray):
    pass

class modelviewmatrix(numpy.ndarray):
    pass

class cameramatrix(numpy.ndarray):
    pass

def matrix(projection, modelview):
    """ Create a camera matrix.
    
        Parameters
        ----------
        projection: projectionmatrix
           from ortho or persp

        modelview : modelviewmatrix
           from lookat

        Returns
        -------
        cameramatrix
    """
    assert isinstance(projection, projectionmatrix)
    assert isinstance(modelview, modelviewmatrix)

    matrix = projection.dot(modelview).view(type=cameramatrix)
    matrix.near = projection.near
    matrix.far = projection.far
    matrix.up = modelview.up
    matrix.side = modelview.side

    return matrix

def ortho(near, far, extent):
    """ An orthogonal camera projection.
        
        Parameters
        ----------
        near : float
            location of the near field

        far : float
            location of the far field
            
        extent : tuple of 4
            left, right, bottom, top in data coordinate

        Returns
        -------
        A projectionmatrix.

    """
    # Input validation
    if near >= far:
        raise ValueError("near must be less than far")
    
    if not isinstance(extent, (list, tuple)) or len(extent) != 4:
        raise ValueError("extent must be a 4-tuple (left, right, bottom, top)")
    
    l, r, b, t = extent
    
    if l >= r:
        raise ValueError("left must be less than right in extent")
    if b >= t:
        raise ValueError("bottom must be less than top in extent")
    ortho = numpy.zeros((4,4))
    ortho[0, 0] = 2.0 / (r - l)
    ortho[1, 1] = 2.0 / (t - b)
    ortho[2, 2] = -2.0 / (far - near)
    ortho[3, 3] = 1
    ortho[0, 3] = - (1. * r + l) / (r - l)
    ortho[1, 3] = - (1. * t + b) / (t - b)
    ortho[2, 3] = - (1. * far + near) / (far - near)
    ortho = ortho.view(type=projectionmatrix)
    ortho.extent = extent
    ortho.near = near
    ortho.far = far
    ortho.scale = numpy.array([ortho[0, 0], ortho[1, 1]])
    return ortho

def fov2extent(fov, aspect, D):
    """ Convert FOV parameters to extent """
    l = - numpy.tan(fov * 0.5) * aspect * D
    r = - l
    b = - numpy.tan(fov * 0.5) * D
    t = - b
    return (l, r, b, t)

def extent2fov(extent, D):
    """ Convert extent to FOV parameters """
    l, r, b, t = extent
    aspect = (r - l) / (t - b)
    fov = numpy.arctan2((t - b), D) * 2
    return fov, aspect

def persp(near, far, fov, aspect):
    """ A perspective camera projection.
        
        Parameters
        ----------
        near : float
            location of the near field

        far : float
            location of the far field

        fov : float
            Field of View angle in radians.

        aspect : float
            aspect ratio. It should be the same as
            the device shape[0] / shape[1] for square pixels.

        Returns
        -------
        A projectionmatrix.

        perspective dot modelview
    """
    # Input validation
    if near >= far:
        raise ValueError("near must be less than far")
    
    if near <= 0:
        raise ValueError("near must be positive")
    
    if fov <= 0 or fov >= numpy.pi:
        raise ValueError("fov must be between 0 and pi radians")
    
    if aspect <= 0:
        raise ValueError("aspect ratio must be positive")
    
    # distance cancels out
    l, r, b, t = fov2extent(fov, aspect, 1)
    
    persp = numpy.zeros((4,4))
 
    persp[0, 0] = 2. / (r - l)
    persp[1, 1] = 2. / (t - b)
    persp[2, 2] = - (1. *(far + near)) / (far - near)
    persp[2, 3] = - (2. * far * near) / (far - near)
    persp[0, 2] = (r + l) / (r - l)
    persp[1, 2] = (t + b) / (t - b)
    persp[3, 2] = -1
    persp[3, 3] = 0

    persp = persp.view(type=projectionmatrix)
    persp.near = near
    persp.far = far
    persp.scale = numpy.array([persp[0, 0], persp[1, 1]])

    return persp

def lookat(pos, target, up):
    """ Set up a camera position matrix (modelview)

        Parameters
        ----------
        pos : 3-tuple
           Position of the camera

        target : 3-tuple
           Focal point of the camera

        up : 3-tuple
           Up direction of the camera

        Returns
        -------
        modelview matrix. The up and side vector are
        stored as attributes.

        Notes
        -----
        The full camera transformation is 

        m = dot(projection, modelview)

    """
    pos = numpy.asarray(pos)
    target = numpy.asarray(target)
    up = numpy.asarray(up)

    dir = target - pos
    dir = dir / (dir **2).sum() ** 0.5
    side = numpy.cross(dir, up)
    side[:] = side / (side **2).sum() ** 0.5
    up = numpy.cross(side, dir)
    up = up / (up **2).sum() ** 0.5

    m1 = numpy.zeros((4,4))
    m1[0, 0:3] = side
    m1[1, 0:3] = up
    m1[2, 0:3] = -dir
    m1[3, 3] = 1

    tran = numpy.eye(4)
    tran[0:3, 3] = -pos
    m2 = numpy.dot(m1, tran)
    m2 = m2.view(type=modelviewmatrix)
    m2.up = up
    m2.side = side
    return m2

def _apply_worker(i):
    """Worker function for apply() - must be at module level for pickling"""
    chunksize = 1024 * 32
    tmppos = _worker_pos[i:chunksize+i]
    tmpout = _worker_shmout[i:chunksize+i]
    tmp = numpy.empty((len(tmppos), 4), dtype='f8')
    tmp[..., 3] = 1.0
    tmp[..., :3] = tmppos
    tmp = numpy.dot(tmp, _worker_matrix.T)
    tmpout[..., :] = tmp[..., :3] / tmp[..., 3][..., None]

def apply(matrix, pos, np=None):
    """ 
        apply a camera matrix to data coordinates

        Parameters
        ----------
        matrix : cameramatrix
           created by :func:`matrix`

        pos : array_like
           data coordinates

        Returns
        -------
        clip_pos : array_like
           The position in the clip coordinate.
           The frustum of the camera is in
           (-1, 1) x (-1, 1) x (-1, 1)

    """
    assert isinstance(matrix, cameramatrix)

    global _worker_pos, _worker_matrix, _worker_shmout
    
    _worker_pos = pos
    _worker_matrix = matrix
    _worker_shmout = sharedmem.empty_like(pos)
    chunksize = 1024 * 32
        
    with sharedmem.MapReduce(np=np) as pool:
        pool.map(_apply_worker, range(0, len(pos), chunksize))

    return _worker_shmout

def clip(xc):
    """ Compute the clipping mask from clipping coordinates.
    
        Parameters
        ----------
        xc : array_like
           position in clipping coordinate.
    
        Returns
        -------
        mask : array_like
           True for inside, False for outside.

    """

    return ((xc >= -1) & (xc <= 1)).all(axis=-1)

def _todevice_worker(i):
    """Worker function for todevice() - must be at module level for pickling"""
    chunksize = 1024 * 32
    l, r, b, t = _worker_extent_params
    tmp = (_worker_xc[i:i+chunksize] + 1.0)
    tmp *= 0.5
    tmp[..., 1] *= (t - b)
    tmp[..., 1] += b
    tmp[..., 0] *= (r - l)
    tmp[..., 0] += l
    _worker_out[i:i+chunksize] = tmp[:, :2]

def todevice(xc, extent, np=None):
    """ Convert clipping coordinate to device coordinate.
    
        Parameters
        ----------
        xc : array_like (... ,3)
            Clipping coordinates. Only the first two columns
            of xc are used.
            
        extent : 2-tuple or 4-tuple
           The extent of the device coordinate.
           (E[0], E[1]) or (S[0], E[0], S[1], E[1]).
           In first case S[0] and S[1] are 0.

        Returns
        -------
        xd : array_like (..., 2)
            data in device coordinate.
            For data in the frustum, it shall be
            between S and E. Only the first two columns
            of xd are useful.

    """
    global _worker_xc, _worker_out, _worker_extent_params
    
    if len(extent) == 2:
        r, t = extent
        l, b = 0, 0
    else:
        l, r, b, t = extent

    _worker_xc = xc
    _worker_out = sharedmem.empty((len(xc), 2))
    _worker_extent_params = (l, r, b, t)
    
    chunksize = 1024 * 32
    with sharedmem.MapReduce(np=np) as pool:
        pool.map(_todevice_worker, range(0, len(xc), chunksize))

    return _worker_out

def todevice_shape(xc, shape, np=None):
    """ Convert clipping coordinate to device coordinate using image shape.
    
        This is a more intuitive version of todevice() that uses image shape
        instead of extent. When users specify shape=(height, width), the 
        device coordinates will range from [0, width-1] and [0, height-1].
        
        Parameters
        ----------
        xc : array_like (... ,3)
            Clipping coordinates. Only the first two columns
            of xc are used.
            
        shape : 2-tuple
            The shape of the output image as (height, width).
            Device coordinates will range from [0, width-1] and [0, height-1].

        Returns
        -------
        xd : array_like (..., 2)
            Device coordinates as (x, y) where x is in [0, width-1] 
            and y is in [0, height-1]. For data in the frustum.

    """
    # Input validation
    xc = numpy.asarray(xc)
    if xc.ndim < 2 or xc.shape[-1] < 2:
        raise ValueError("xc must have at least 2 columns in the last dimension")
    
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError("shape must be a 2-tuple (height, width)")
    
    height, width = shape
    if not (isinstance(height, int) and isinstance(width, int)):
        raise ValueError("shape must contain integers")
    
    if height <= 0 or width <= 0:
        raise ValueError("shape must contain positive values")
    
    return todevice(xc, (width-1, height-1), np=np)

def data_to_device(matrix, pos, sml, extent, apply_clip=False, np=None):
    """ Transform pos and smoothing length in device coordinate,
        to device with finite differentiation.
    
        Parameters
        ----------
        matrix: camera matrix

        pos : array_like
            position in data coordinate

        sml : array_like
            smoothing in data coordinate

        extent : 2-tuple or 4-tuple
            (see todevice)

        apply_clip : bool
            if True, the clipping will be applied
        Returns
        -------
        xd, smld, clip : array_like
            position and sml in device coordinate, and clipping flag.

            xd and smld are clipped by clip if apply_clip is True.

    """
    assert isinstance(matrix, cameramatrix)

    xc = apply(matrix, pos + sml[:, None] * matrix.side, np=np)

    m = clip(xc)
    if apply_clip:
        xc = xc[m]
        pos = pos[m]
        sml = sml[m]

    xd = todevice(xc, extent, np=np)

    xc1 = apply(matrix, pos + sml[:, None] * matrix.side, np=np)
    xd1 = todevice(xc1, extent, np=np)

    smld = ((xd1 - xd) ** 2).sum(axis=-1) ** 0.5
    return xd, smld, m

if __name__ == '__main__':
    test()
