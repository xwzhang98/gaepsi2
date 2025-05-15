"""
Camera module for gaepsi2.

This module provides OpenGL-like matrix transformations to project 3D particle
data to 2D screen coordinates. The camera pipeline mimics the OpenGL rendering
pipeline with projection and model-view matrices.

The typical camera transformation pipeline:
1. Data coordinates (3D positions in simulation space)
2. Clip coordinates (after applying camera matrix)
3. Device coordinates (2D screen positions)

Main functions:
- matrix: Create a camera matrix from projection and modelview matrices
- ortho: Create an orthographic projection matrix
- persp: Create a perspective projection matrix
- lookat: Create a modelview matrix from camera position, target and up vector
- apply: Apply camera transformation to points
- clip: Determine which points are within the viewing frustum
- todevice: Convert clip coordinates to device (screen) coordinates
- data_to_device: Full pipeline transformation from data to device coordinates
"""

import numpy as np  # Import NumPy for array operations
import sharedmem    # Import sharedmem for parallel processing
from typing import Tuple, Union, Optional, List, Any  # Import type hints

class projectionmatrix(np.ndarray):
    """Matrix representing a projection transformation.

    This subclass of ndarray stores additional attributes specific to projection matrices.
    """
    pass

class modelviewmatrix(np.ndarray):
    """Matrix representing a modelview transformation.

    This subclass of ndarray stores additional attributes specific to modelview matrices.
    """
    pass

class cameramatrix(np.ndarray):
    """Combined camera matrix (projection * modelview).

    This subclass of ndarray stores additional attributes specific to camera matrices.
    """
    pass

def matrix(projection: projectionmatrix,
           modelview: modelviewmatrix) -> cameramatrix:
    """Create a camera matrix by combining projection and modelview matrices.

    Parameters
    ----------
    projection : projectionmatrix
        Projection matrix from ortho or persp
    modelview : modelviewmatrix
        Modelview matrix from lookat

    Returns
    -------
    cameramatrix
        Combined camera matrix
    """
    # Verify the input matrices are of the correct type
    assert isinstance(projection, projectionmatrix)
    assert isinstance(modelview, modelviewmatrix)

    # Multiply projection and modelview matrices to create the camera matrix
    # The .dot() method performs matrix multiplication
    # .view() changes the type to cameramatrix while preserving the data
    matrix = projection.dot(modelview).view(type=cameramatrix)

    # Copy attributes from input matrices to the combined matrix
    matrix.near = projection.near  # Near clipping plane distance
    matrix.far = projection.far    # Far clipping plane distance
    matrix.up = modelview.up       # Up vector from modelview
    matrix.side = modelview.side   # Side vector from modelview

    return matrix

def ortho(near: float,
          far: float,
          extent: Tuple[float, float, float, float]) -> projectionmatrix:
    """Create an orthographic projection matrix.

    Parameters
    ----------
    near : float
        Location of the near clipping plane
    far : float
        Location of the far clipping plane
    extent : tuple of 4 floats
        (left, right, bottom, top) in data coordinates

    Returns
    -------
    projectionmatrix
        Orthographic projection matrix
    """
    # Unpack the extent parameters
    l, r, b, t = extent  # left, right, bottom, top boundaries

    # Create a 4x4 matrix filled with zeros
    ortho = np.zeros((4, 4))

    # Set the diagonal elements for x and y scaling
    ortho[0, 0] = 2.0 / (r - l)  # X-axis scaling factor
    ortho[1, 1] = 2.0 / (t - b)  # Y-axis scaling factor

    # Set the z-scaling factor (negative because OpenGL uses right-handed coord system)
    ortho[2, 2] = -2.0 / (far - near)

    # Set the homogeneous coordinate scaling
    ortho[3, 3] = 1

    # Set the translation components
    ortho[0, 3] = - (1. * r + l) / (r - l)  # X-axis translation
    ortho[1, 3] = - (1. * t + b) / (t - b)  # Y-axis translation
    ortho[2, 3] = - (1. * far + near) / (far - near)  # Z-axis translation

    # Convert the matrix to projectionmatrix type
    ortho = ortho.view(type=projectionmatrix)

    # Store additional attributes
    ortho.extent = extent  # Keep the original extent
    ortho.near = near      # Store near clipping plane
    ortho.far = far        # Store far clipping plane
    ortho.scale = np.array([ortho[0, 0], ortho[1, 1]])  # Store scaling factors

    return ortho

def fov2extent(fov: float, aspect: float, D: float) -> Tuple[float, float, float, float]:
    """Convert field of view parameters to extent.

    Parameters
    ----------
    fov : float
        Field of view in radians
    aspect : float
        Aspect ratio (width/height)
    D : float
        Distance to the projection plane

    Returns
    -------
    tuple
        (left, right, bottom, top) extent
    """
    # Calculate the half-width based on field of view, aspect ratio and distance
    l = - np.tan(fov * 0.5) * aspect * D  # Left boundary
    r = - l  # Right boundary (symmetric about center)

    # Calculate the half-height based on field of view and distance
    b = - np.tan(fov * 0.5) * D  # Bottom boundary
    t = - b  # Top boundary (symmetric about center)

    return (l, r, b, t)  # Return the calculated extent

def extent2fov(extent: Tuple[float, float, float, float],
               D: float) -> Tuple[float, float]:
    """Convert extent to field of view parameters.

    Parameters
    ----------
    extent : tuple of 4 floats
        (left, right, bottom, top) in data coordinates
    D : float
        Distance to the projection plane

    Returns
    -------
    tuple
        (fov, aspect) field of view and aspect ratio
    """
    # Unpack the extent parameters
    l, r, b, t = extent  # left, right, bottom, top boundaries

    # Calculate the aspect ratio from the extent
    aspect = (l - r) /(b - t)  # Width to height ratio

    # Calculate the field of view from the height and distance
    fov = np.arctan2((t - b), D) * 2  # Field of view angle in radians

    return fov, aspect  # Return calculated FOV and aspect ratio

def persp(near: float,
          far: float,
          fov: float,
          aspect: float) -> projectionmatrix:
    """Create a perspective projection matrix.

    Parameters
    ----------
    near : float
        Location of the near clipping plane
    far : float
        Location of the far clipping plane
    fov : float
        Field of view in radians
    aspect : float
        Aspect ratio (width/height)

    Returns
    -------
    projectionmatrix
        Perspective projection matrix
    """
    # Convert FOV to extent at distance 1 (arbitrary, distances cancel out)
    l, r, b, t = fov2extent(fov, aspect, 1)

    # Create a 4x4 matrix filled with zeros
    persp = np.zeros((4, 4))

    # Set the x and y scaling factors
    persp[0, 0] = 2. / (r - l)  # X-axis scaling
    persp[1, 1] = 2. / (t - b)  # Y-axis scaling

    # Set z-related parameters for perspective correction
    persp[2, 2] = - (1. *(far + near)) / (far - near)  # Z-axis scaling
    persp[2, 3] = - (2. * far * near) / (far - near)   # Z-axis translation

    # Set parameters for perspective division
    persp[0, 2] = (r + l) / (r - l)  # X component of direction vector
    persp[1, 2] = (t + b) / (t - b)  # Y component of direction vector
    persp[3, 2] = -1  # Set w-coordinate for perspective division
    persp[3, 3] = 0   # Required for proper perspective division

    # Convert the matrix to projectionmatrix type
    persp = persp.view(type=projectionmatrix)

    # Store additional attributes
    persp.near = near  # Store near clipping plane
    persp.far = far    # Store far clipping plane
    persp.scale = np.array([persp[0, 0], persp[1, 1]])  # Store scaling factors

    return persp

def lookat(pos: Tuple[float, float, float],
           target: Tuple[float, float, float],
           up: Tuple[float, float, float]) -> modelviewmatrix:
    """Create a modelview matrix from camera position, target and up vector.

    Parameters
    ----------
    pos : tuple of 3 floats
        Position of the camera
    target : tuple of 3 floats
        Point the camera is looking at
    up : tuple of 3 floats
        Up direction vector for the camera

    Returns
    -------
    modelviewmatrix
        Modelview matrix
    """
    # Convert inputs to numpy arrays for vector operations
    pos = np.asarray(pos)      # Camera position
    target = np.asarray(target)  # Target point
    up = np.asarray(up)          # Up vector

    # Calculate the camera's forward direction vector (from camera to target)
    dir = target - pos  # Direction vector
    dir = dir / np.sqrt(np.sum(dir**2))  # Normalize to unit length

    # Calculate the camera's right vector (perpendicular to dir and up)
    side = np.cross(dir, up)  # Cross product gives perpendicular vector
    side = side / np.sqrt(np.sum(side**2))  # Normalize to unit length

    # Recalculate up vector to ensure it's perpendicular to both dir and side
    up = np.cross(side, dir)  # Cross product gives perpendicular vector
    up = up / np.sqrt(np.sum(up**2))  # Normalize to unit length

    # Create the rotation part of the modelview matrix
    m1 = np.zeros((4, 4))
    m1[0, 0:3] = side   # First row is the side (right) vector
    m1[1, 0:3] = up     # Second row is the up vector
    m1[2, 0:3] = -dir   # Third row is the negative dir vector (camera looks along -Z)
    m1[3, 3] = 1        # Homogeneous coordinate scaling

    # Create the translation part of the modelview matrix
    tran = np.eye(4)  # Identity matrix
    tran[0:3, 3] = -pos  # Translation vector (negative camera position)

    # Combine rotation and translation (rotation comes first)
    m2 = np.dot(m1, tran)

    # Convert to modelviewmatrix type
    m2 = m2.view(type=modelviewmatrix)

    # Store additional attributes
    m2.up = up      # Store the calculated up vector
    m2.side = side  # Store the calculated side vector
    return m2

def apply(matrix: cameramatrix,
          pos: np.ndarray,
          np: Optional[int] = None) -> np.ndarray:
    """Apply camera transformation to data coordinates.

    Parameters
    ----------
    matrix : cameramatrix
        Camera matrix from matrix()
    pos : array_like
        Data coordinates, shape (..., 3)
    np : int, optional
        Number of processes for parallel computation, or None for auto

    Returns
    -------
    array_like
        Clip coordinates, shape (..., 3)
    """
    # Verify the input matrix is of the correct type
    assert isinstance(matrix, cameramatrix)

    # Create shared memory output array with same shape as input
    shmout = sharedmem.empty_like(pos)
    chunksize = 1024 * 32  # Process data in chunks for better performance

    # Define the transformation function to apply to each chunk
    def work(i):
        # Extract a chunk of positions
        tmppos = pos[i:chunksize+i]
        tmpout = shmout[i:chunksize+i]

        # Create a temporary array with homogeneous coordinates (add w=1)
        tmp = np.empty((len(tmppos), 4), dtype='f8')
        tmp[..., 3] = 1.0  # Set w coordinate to 1
        tmp[..., :3] = tmppos  # Copy position coordinates

        # Apply the camera matrix transformation
        tmp = np.dot(tmp, matrix.T)  # Matrix multiplication

        # Perform perspective division to get clip coordinates
        tmpout[..., :] = tmp[..., :3] / tmp[..., 3][..., None]

    # Use sharedmem for parallel processing
    with sharedmem.MapReduce(np=np) as pool:
        pool.map(work, range(0, len(pos), chunksize))

    return shmout  # Return the transformed coordinates

def clip(xc: np.ndarray) -> np.ndarray:
    """Compute the clipping mask from clipping coordinates.

    Parameters
    ----------
    xc : array_like
        Positions in clipping coordinates, shape (..., 3)

    Returns
    -------
    array_like
        Boolean mask, True for points inside the viewing frustum
    """
    # Check if all coordinates are within the [-1, 1] range for all dimensions
    # This defines the canonical view volume (viewing frustum in clip coordinates)
    return ((xc >= -1) & (xc <= 1)).all(axis=-1)

def todevice(xc: np.ndarray,
             extent: Union[Tuple[float, float], Tuple[float, float, float, float]],
             np: Optional[int] = None) -> np.ndarray:
    """Convert clipping coordinates to device coordinates.

    Parameters
    ----------
    xc : array_like
        Clipping coordinates, shape (..., 3)
    extent : tuple
        Either (width, height) or (left, right, bottom, top)
    np : int, optional
        Number of processes for parallel computation, or None for auto

    Returns
    -------
    array_like
        Device coordinates, shape (..., 2)
    """
    # Determine the device extent based on input
    if len(extent) == 2:
        r, t = extent  # Width and height only
        l, b = 0, 0    # Default left and bottom to 0
    else:
        l, r, b, t = extent  # Full extent specification

    # Set up for parallel processing
    chunksize = 1024 * 32  # Process data in chunks
    out = sharedmem.empty((len(xc), 2))  # Output array for device coordinates

    # Define the transformation function for each chunk
    def work(i):
        # Extract a chunk of clip coordinates
        tmp = (xc[i:i+chunksize] + 1.0)  # Map from [-1,1] to [0,2]
        tmp *= 0.5  # Map from [0,2] to [0,1]

        # Scale Y-coordinate to device height and add bottom offset
        tmp[..., 1] *= (t - b)
        tmp[..., 1] += b

        # Scale X-coordinate to device width and add left offset
        tmp[..., 0] *= (r - l)
        tmp[..., 0] += l

        # Store result (only keep X and Y coordinates)
        out[i:i+chunksize] = tmp[:, :2]

    # Use sharedmem for parallel processing
    with sharedmem.MapReduce(np=np) as pool:
        pool.map(work, range(0, len(xc), chunksize))

    return out  # Return the device coordinates

def data_to_device(matrix: cameramatrix,
                  pos: np.ndarray,
                  sml: np.ndarray,
                  extent: Union[Tuple[float, float], Tuple[float, float, float, float]],
                  apply_clip: bool = False,
                  np: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform positions and smoothing lengths from data to device coordinates.

    Parameters
    ----------
    matrix : cameramatrix
        Camera matrix from matrix()
    pos : array_like
        Data coordinates, shape (..., 3)
    sml : array_like
        Smoothing lengths in data coordinates
    extent : tuple
        Either (width, height) or (left, right, bottom, top)
    apply_clip : bool, optional
        If True, points outside the viewing frustum are clipped
    np : int, optional
        Number of processes for parallel computation, or None for auto

    Returns
    -------
    xd : array_like
        Device coordinates, shape (..., 2)
    smld : array_like
        Smoothing lengths in device coordinates
    mask : array_like
        Boolean mask, True for points inside the viewing frustum
    """
    # Verify the input matrix is of the correct type
    assert isinstance(matrix, cameramatrix)

    # Apply the camera transformation to positions slightly perturbed by smoothing length
    # This allows us to estimate the size of a particle in screen space
    xc = apply(matrix, pos + sml[:, None] * matrix.side, np=np)

    # Compute clipping mask to identify particles inside the viewing frustum
    m = clip(xc)

    # If apply_clip is True, filter out particles outside the viewing frustum
    if apply_clip:
        xc = xc[m]
        pos = pos[m]
        sml = sml[m]

    # Convert clip coordinates to device coordinates
    xd = todevice(xc, extent, np=np)

    # Calculate another point at the edge of each particle
    # This is used to determine the particle size in device coordinates
    xc1 = apply(matrix, pos + sml[:, None] * matrix.side, np=np)
    xd1 = todevice(xc1, extent, np=np)

    # Calculate the smoothing length in device coordinates
    # This is the distance between the particle center and edge in screen space
    smld = np.sqrt(np.sum((xd1 - xd) ** 2, axis=-1))

    return xd, smld, m  # Return device coordinates, device smoothing lengths, and clipping mask