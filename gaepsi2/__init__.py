"""
gaepsi2: SPH visualization package

A suite of routines for visualizing SPH (Smoothed Particle Hydrodynamics) simulations.
Provides camera transformations, SPH resampling, color mapping, and cosmological utilities.
"""

from .version import __version__

# Core camera functionality
from .camera import (
    matrix, ortho, persp, lookat, apply, clip, 
    todevice, todevice_shape, data_to_device,
    fov2extent, extent2fov
)

# Painting/rendering
from .painter import paint

# Color mapping
from .color import N, NL, CoolWarm, Hot, Colormap

# Cosmology utilities  
from .cosmology import WMAP7

# Survey volume remapping
from . import svr

__all__ = [
    '__version__',
    # Camera functions
    'matrix', 'ortho', 'persp', 'lookat', 'apply', 'clip',
    'todevice', 'todevice_shape', 'data_to_device',
    'fov2extent', 'extent2fov',
    # Painting
    'paint', 
    # Color mapping
    'N', 'NL', 'CoolWarm', 'Hot', 'Colormap',
    # Cosmology
    'WMAP7',
    # Survey volume remapping
    'svr'
]
