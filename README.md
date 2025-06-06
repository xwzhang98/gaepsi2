# gaepsi2

A Python package for visualizing Smoothed Particle Hydrodynamics (SPH) simulations with high-performance painting and camera transformations.

## Features

- **SPH Painting**: Fast SPH kernel-based particle painting to 2D images
- **Camera Transformations**: OpenGL-style matrix transformations for 3D to 2D projection
- **Periodic Boundaries**: Support for periodic boundary conditions in simulations
- **Multi-channel Painting**: Paint multiple physical quantities simultaneously
- **Parallel Processing**: Multi-core support for large datasets
- **Efficient Colormaps**: Optimized color mapping for large images

## Installation

### From Source (Development)

```bash
git clone https://github.com/xwzhang98/gaepsi2.git
cd gaepsi2
pip install -e .
```

### Dependencies

- Python >= 3.7
- NumPy >= 1.20
- Cython >= 0.29
- scipy
- sharedmem

## Quick Start

### Basic SPH Painting

```python
import numpy as np
from gaepsi2 import painter

# Create particles in world coordinates
n_particles = 1000
world_size = 64.0
pos_world = np.random.uniform(0, world_size, (n_particles, 2))  # 2D positions
sml_world = np.full(n_particles, 2.0)  # smoothing lengths in world units
density = np.random.uniform(0.5, 2.0, n_particles)  # particle properties

# Convert from world coordinates to device (pixel) coordinates
image_shape = (256, 256)
scale = image_shape[0] / world_size  # pixels per world unit
pos_device = pos_world * scale
sml_device = sml_world * scale

# Paint to image
result = painter.paint(pos_device, sml_device, [density], image_shape)
density_image = result[0]

# For visualization, remember to transpose
import matplotlib.pyplot as plt
plt.imshow(density_image.T, origin='lower', extent=[0, world_size, 0, world_size], cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('X (world units)')
plt.ylabel('Y (world units)')
plt.show()
```

### 3D Camera Projection

```python
from gaepsi2 import camera, painter

# 3D particle positions
pos_3d = np.random.uniform(-10, 10, (n_particles, 3))

# Set up camera
proj = camera.ortho(near=1, far=100, extent=(-15, 15, -15, 15))
mv = camera.lookat(pos=[0, 0, -50], target=[0, 0, 0], up=[0, 1, 0])
cam_matrix = camera.matrix(proj, mv)

# Transform to 2D
xc = camera.apply(cam_matrix, pos_3d)
image_shape = (512, 512)
xd = camera.todevice_shape(xc, image_shape)

# Paint
result = painter.paint(xd, sml, [density], image_shape)
```

### Periodic Boundary Conditions

```python
# For simulations with periodic boundaries
result = painter.paint(pos, sml, [density], image_shape, periodic=True)
# Particles near edges will wrap around and contribute to opposite sides
```

## Examples

See `test_plot.py` for comprehensive examples including:
- Basic SPH painting
- 3D camera transformations
- Perspective projections
- Color normalization
- Cosmological visualizations
- Periodic boundary demonstrations