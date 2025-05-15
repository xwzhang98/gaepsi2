# README.md

# Gaepsi2

A suite of routines for visualizing SPH (Smoothed Particle Hydrodynamics) simulations in astrophysics.

[![PyPI version](https://badge.fury.io/py/gaepsi2.svg)](https://badge.fury.io/py/gaepsi2)
[![Build Status](https://github.com/rainwoodman/gaepsi2/actions/workflows/test.yml/badge.svg)](https://github.com/rainwoodman/gaepsi2/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/gaepsi2/badge/?version=latest)](https://gaepsi2.readthedocs.io/en/latest/?badge=latest)

## Features

- **Camera**: OpenGL-alike matrix transformations from simulation to clipping and device coordinates
- **SVR**: Survey volume remapping for cosmological simulations
- **Painter**: SPH resampling windows for efficient particle visualization
- **Color**: Efficient colormaps for large images

## Installation

### From PyPI

```bash
pip install gaepsi2
```

### From source

```bash
git clone https://github.com/rainwoodman/gaepsi2.git
cd gaepsi2
pip install -e .
```

## Quick Start

Here's a simple example to get you started:

```python
import numpy as np
from gaepsi2 import camera, painter

# Create some test particles
pos = np.random.uniform(size=(1000, 3)) * 20.
pos -= 10.

# Set up camera
proj = camera.ortho(0, 20, (-10, 10, -10, 10))
mv = camera.lookat((0, 0, -10), (0, 0, 0), (0, 1, 0))
matrix = camera.matrix(proj, mv)

# Project positions to 2D
p2d = camera.apply(matrix, pos)

# Create smoothing lengths and data values
sml = np.ones(len(pos)) * 0.5
data = np.ones(len(pos))

# Create image
image = painter.paint(p2d[:, :2], sml, [data], (400, 400))

# Visualize with matplotlib
import matplotlib.pyplot as plt
plt.imshow(image[0], origin='lower')
plt.colorbar()
plt.show()
```

## Documentation

For detailed documentation, see [the documentation site](https://gaepsi2.readthedocs.io/).

## API Overview

### Camera Module

- `camera.ortho`: Create an orthographic projection
- `camera.persp`: Create a perspective projection
- `camera.lookat`: Set up a camera position matrix
- `camera.matrix`: Combine projection and model view matrices
- `camera.apply`: Apply camera transformation to positions
- `camera.data_to_device`: Transform data space to device space

### SVR Module

- `svr.remap`: Remap a cube to a sheet (useful for cosmological simulations)
- `svr.wrap`: Convert positions in a box to [0, 1] range

### Painter Module

- `painter.paint`: Paint particles to an image using SPH kernel interpolation

### Color Module

- `color.CoolWarm`: Efficient colormap for large images
- `color.N`: Normalize array values to [0, 1]
- `color.NL`: Normalize array values in log scale to [0, 1]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 2-Clause License - see the LICENSE file for details.

## Citation

If you use gaepsi2 in a scientific publication, we would appreciate a citation to the GitHub repository.

## Acknowledgments

- The SPH visualization algorithms are based on research in computational astrophysics
- Thanks to all contributors and users of the library
