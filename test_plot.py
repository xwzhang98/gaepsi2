import numpy as np
import matplotlib.pyplot as plt
from gaepsi2 import camera, painter


p2d = np.array([
    [5, 5],
    [15, 15],
    [5, 15]
])
# Create smoothing lengths and data values
sml = np.ones(len(p2d)) * 2.0
data = np.ones(len(p2d))

# Create image
image = painter.paint(p2d, sml, [data], (20, 20))

# Visualize with matplotlib
plt.imshow(image[0].T, origin='lower')
plt.colorbar()
plt.savefig('sml_large.jpg', format='jpg') # Have to save before show
plt.close()


p2d = np.array([
    [5, 5],
    [15, 15],
    [5, 15]
])
print(p2d.shape)
# Create smoothing lengths and data values
sml = np.ones(len(p2d)) * 0.2
data = np.ones(len(p2d))

# Create image
image = painter.paint(p2d, sml, [data], (20, 20))

# Visualize with matplotlib
plt.imshow(image[0].T, origin='lower')
plt.colorbar()
plt.savefig('sml_small.jpg', format='jpg') # Have to save before show
plt.close()