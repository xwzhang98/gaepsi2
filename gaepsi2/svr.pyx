"""
Survey Volume Remapping in Gaepsi2

This module provides Cython implementations for remapping a cube to a sheet,
which is particularly useful for visualizing cosmological simulations.

The core functionality:
- wrap: Convert positions in a box to the [0, 1] range
- remap: Apply a 3x3 transformation matrix to coordinates
- remap_query_size: Get the dimensions of the remapped volume

The implementation uses C functions from svremap.c and transform.c for performance.
"""

# Import NumPy for Python array operations
import numpy
# Import NumPy's C API for efficient array manipulation in Cython
cimport numpy

# Import Cython for type declarations and optimizations
import cython
cimport cython

# Import C functions from external C files
# The 'pass' statement indicates we're not importing specific symbols from transform.c
cdef extern from 'c/transform.c':
    pass

# Import the SVRemap struct and related functions from svremap.c
cdef extern from 'c/svremap.c':
    # Define the SVRemap struct with a size field (other fields are hidden with 'pass')
    ctypedef struct SVRemap "SVRemap":
        double size[3]  # Size of the remapped volume
        pass

    # Function to initialize an SVRemap struct with a transformation matrix
    int svremap_init(SVRemap * r, int * remap)

    # Function to apply the remapping to coordinates
    double svremap_apply(SVRemap * r, double x[3], double y[3], int I[3])

def wrap(pos, BoxSize, offset=None, output=None):
    """ Wrap positions in a box of BoxSize into (0, 1) range.

    Parameters
    ----------
    pos : array_like
        Position array with shape [..., Nd]
    BoxSize : float or array_like
        Size of the box
    offset : float or array_like, optional
        Offset to apply before normalization
    output : array_like, optional
        Pre-allocated output array

    Returns
    -------
    array_like
        Wrapped positions in range [0, 1]
    """
    # Create a normalization factor by dividing 1.0 by BoxSize
    norm = numpy.empty(pos.shape[-1])
    norm[:] = BoxSize
    norm[:] = 1.0 / norm[:]

    # Allocate output array if not provided
    if output is None:
        output = numpy.empty_like(pos)

    # Apply offset if provided
    if offset and offset != 0.0:
        off = numpy.empty(pos.shape[-1])
        off[:] = offset
        pos = numpy.substract(pos, off, output)

    # Normalize positions by BoxSize
    numpy.multiply(pos, norm, output)

    # Apply modulo 1.0 to wrap into [0, 1] range
    return numpy.remainder(output, 1.0, output)

def remap_query_size(M):
    """ Query the size of the remapped volume.

    Parameters
    ----------
    M : array_like
        3x3 integer transformation matrix

    Returns
    -------
    array_like
        Size of the remapped volume
    """
    # Declare C variables
    cdef SVRemap r
    cdef int[:, ::1] matrix
    cdef double[::1] size

    # Convert Python matrix to C-compatible format
    matrix = numpy.array(M, dtype='int32')

    # Initialize the SVRemap struct with the matrix
    svremap_init(&r, &matrix[0, 0])

    # Extract and return the size from the SVRemap struct
    return numpy.array(<double [:3]> r.size).copy()

# Apply Cython optimization decorators to disable runtime checks for performance
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.overflowcheck(False)
@cython.nonecheck(False)
def remap(pos, M, output=None):
    """ Apply survey volume remapping to coordinates.

    Parameters
    ----------
    pos : array_like
        Position array with shape [..., 3] and values in range [0, 1]
    M : array_like
        3x3 integer transformation matrix
    output : array_like, optional
        Pre-allocated output array

    Returns
    -------
    array_like
        Remapped positions

    Notes
    -----
    The input positions must be in range [0, 1] for all dimensions.
    No sanity checks are performed for performance reasons.
    """
    # Allocate output array if not provided
    if output is None:
        output = numpy.empty_like(pos, dtype='f8')
    else:
        # Ensure output has the correct data type
        assert output.dtype == numpy.dtype('f8')

    # Ensure output has the correct shape
    assert output.shape[output.ndim - 1] == 3

    # Copy input to output (will be modified in place)
    output[...] = pos

    # Declare Cython typed variables for efficient access
    cdef int[:, ::1] matrix     # The transformation matrix
    cdef int I[3]               # Integer coordinates
    cdef double[:, ::1] x = output  # Input array view
    cdef double[:, ::1] y = output  # Output array view (same as input for in-place operation)
    cdef double tmp[3]          # Temporary buffer for remapped coordinates
    cdef SVRemap r              # SVRemap structure

    # Convert Python matrix to C-compatible format
    matrix = numpy.array(M, dtype='int32')

    # Initialize integer coordinates to zero
    I[0] = 0
    I[1] = 0
    I[2] = 0

    # Initialize the SVRemap struct with the matrix
    svremap_init(&r, &matrix[0, 0])

    # Loop through all positions
    cdef numpy.intp_t i
    for i in range(x.shape[0]):
        # Apply the remapping to this position
        svremap_apply(&r, &x[i, 0], &tmp[0], I)

        # Copy the remapped coordinates back to the output array
        y[i, 0] = tmp[0]
        y[i, 1] = tmp[1]
        y[i, 2] = tmp[2]

    return output