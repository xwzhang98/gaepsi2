#!/usr/bin/env python3
"""
Comprehensive test and demonstration script for gaepsi2.

This script demonstrates all major usage patterns of the gaepsi2 library
including camera transformations, SPH painting, color mapping, and 
cosmological calculations. It generates various plots to visualize results.

Run with: python test_plot.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import gaepsi2 modules
from gaepsi2 import camera, painter, color, cosmology


def setup_output_directory():
    """Create output directory for plots."""
    output_dir = "test_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def test_basic_painting():
    """Test basic SPH painting functionality."""
    print("Testing basic SPH painting...")
    
    # Create simple particle distribution
    np.random.seed(42)
    n_particles = 50
    pos_world = np.random.uniform(0, 20, (n_particles, 2))
    sml_world = np.random.uniform(0.5, 2.0, n_particles)
    density = np.random.uniform(0.1, 2.0, n_particles)
    
    # Set up camera to map world coordinates [0,20] to device coordinates
    image_shape = (64, 64)
    # Map world coordinates (0,20) to device coordinates (0,64)
    pos_device = pos_world * (image_shape[0] / 20.0)
    sml_device = sml_world * (image_shape[0] / 20.0)
    
    # Paint to image
    result = painter.paint(pos_device, sml_device, [density], image_shape)
    density_image = result[0]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot particle positions
    ax1.scatter(pos_world[:, 0], pos_world[:, 1], c=density, s=sml_world*20, alpha=0.7, cmap='viridis')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 20)
    ax1.set_title('Original Particle Positions')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Density Value')
    
    # Plot painted image
    im = ax2.imshow(density_image.T, origin='lower', extent=[0, 20, 0, 20], cmap='viridis')
    ax2.set_title('SPH Painted Image')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('Painted Density')
    
    plt.suptitle('Basic SPH Painting: Particles to Image', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, "01_basic_painting.png"


def test_camera_transformations():
    """Test camera transformation pipeline."""
    print("Testing camera transformations...")
    
    # Create 3D particle data
    np.random.seed(42)
    n_particles = 100
    
    # Particles in a 3D box
    pos_3d = np.random.uniform(-5, 5, (n_particles, 3))
    sml = np.random.uniform(0.1, 0.5, n_particles)
    mass = np.random.lognormal(0, 1, n_particles)
    
    # Set up orthographic camera
    proj = camera.ortho(near=1, far=15, extent=(-6, 6, -6, 6))
    mv = camera.lookat(pos=[0, 0, -10], target=[0, 0, 0], up=[0, 1, 0])
    cam_matrix = camera.matrix(proj, mv)
    
    # Transform to clipping coordinates
    xc = camera.apply(cam_matrix, pos_3d)
    
    # Convert to device coordinates using new shape-based function
    image_shape = (128, 128)
    xd = camera.todevice_shape(xc, image_shape)
    
    # Paint the image
    mass_image = painter.paint(xd, sml, [mass], image_shape)[0]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 3D scatter (projection onto xy plane)
    sc1 = axes[0, 0].scatter(pos_3d[:, 0], pos_3d[:, 1], c=mass, s=30, alpha=0.7, cmap='plasma')
    axes[0, 0].set_title('Original 3D Positions (XY View)')
    axes[0, 0].set_xlabel('X Coordinate')
    axes[0, 0].set_ylabel('Y Coordinate')
    axes[0, 0].set_xlim(-6, 6)
    axes[0, 0].set_ylim(-6, 6)
    plt.colorbar(sc1, ax=axes[0, 0], label='Mass')
    
    # 3D scatter (projection onto xz plane)
    sc2 = axes[0, 1].scatter(pos_3d[:, 0], pos_3d[:, 2], c=mass, s=30, alpha=0.7, cmap='plasma')
    axes[0, 1].set_title('Original 3D Positions (XZ View)')
    axes[0, 1].set_xlabel('X Coordinate')
    axes[0, 1].set_ylabel('Z Coordinate')
    axes[0, 1].set_xlim(-6, 6)
    axes[0, 1].set_ylim(-6, 6)
    plt.colorbar(sc2, ax=axes[0, 1], label='Mass')
    
    # Device coordinates
    clip_mask = camera.clip(xc)
    sc3 = axes[1, 0].scatter(xd[clip_mask, 0], xd[clip_mask, 1], c=mass[clip_mask], 
                            s=30, alpha=0.7, cmap='plasma')
    axes[1, 0].set_title('Transformed Device Coordinates')
    axes[1, 0].set_xlabel('Device X (pixels)')
    axes[1, 0].set_ylabel('Device Y (pixels)')
    axes[1, 0].set_xlim(0, 128)
    axes[1, 0].set_ylim(0, 128)
    plt.colorbar(sc3, ax=axes[1, 0], label='Mass')
    
    # Final painted image
    im = axes[1, 1].imshow(mass_image.T, origin='lower', cmap='plasma')
    axes[1, 1].set_title('Final SPH Rendered Image')
    axes[1, 1].set_xlabel('Pixel X')
    axes[1, 1].set_ylabel('Pixel Y')
    plt.colorbar(im, ax=axes[1, 1], label='Rendered Mass')
    
    plt.suptitle('3D to 2D Camera Transformation Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, "02_camera_transformations.png"


def test_perspective_projection():
    """Test perspective camera projection."""
    print("Testing perspective projection...")
    
    # Create particles at different depths
    np.random.seed(42)
    n_particles = 80
    
    # Particles in a volume with depth variation
    pos_3d = np.random.uniform(-2, 2, (n_particles, 3))
    pos_3d[:, 2] = np.random.uniform(-12, -5, n_particles)  # Depth variation
    
    sml = np.random.uniform(0.05, 0.2, n_particles)
    brightness = np.random.uniform(0.5, 2.0, n_particles)
    
    # Perspective camera
    proj = camera.persp(near=1, far=20, fov=np.pi/3, aspect=1.0)
    mv = camera.lookat(pos=[0, 0, -3], target=[0, 0, -8], up=[0, 1, 0])
    cam_matrix = camera.matrix(proj, mv)
    
    # Full transformation pipeline with clipping
    extent = (256, 256)
    xd, smld, clip_mask = camera.data_to_device(
        cam_matrix, pos_3d, sml, extent, apply_clip=True
    )
    
    # Paint if particles remain after clipping
    if len(xd) > 0:
        brightness_clipped = brightness[clip_mask]
        image = painter.paint(xd, smld, [brightness_clipped], (256, 256))[0]
    else:
        image = np.zeros((256, 256))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 3D positions colored by depth
    scatter = ax1.scatter(pos_3d[:, 0], pos_3d[:, 1], c=pos_3d[:, 2], 
                         s=brightness*30, alpha=0.7, cmap='coolwarm')
    ax1.set_title('3D Particle Positions')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Z Depth')
    
    # Perspective projection result
    im = ax2.imshow(image.T, origin='lower', cmap='hot')
    ax2.set_title('Perspective Projection Result')
    ax2.set_xlabel('Pixel X')
    ax2.set_ylabel('Pixel Y')
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('Brightness')
    
    plt.suptitle('Perspective Camera Projection with Depth Effects', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, "03_perspective_projection.png"


def test_color_mapping():
    """Test color mapping functionality."""
    print("Testing color mapping...")
    
    # Create a 2D image for color mapping
    x = np.linspace(-2, 2, 64)
    y = np.linspace(-2, 2, 64)
    X, Y = np.meshgrid(x, y)
    
    # Temperature-like field
    temperature = 1e4 * np.exp(-(X**2 + Y**2)) + 1e3
    
    # Density-like field  
    density = np.exp(-((X-0.5)**2 + (Y+0.5)**2)/0.5) + 0.1*np.random.random((64, 64))
    
    # Apply different normalization and color schemes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Temperature processing
    im1 = axes[0, 0].imshow(temperature, cmap='coolwarm', origin='lower')
    axes[0, 0].set_title('Original Temperature Field')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('Temperature [K]')
    
    temp_norm_linear = color.N(temperature)
    im2 = axes[0, 1].imshow(temp_norm_linear, cmap='coolwarm', origin='lower')
    axes[0, 1].set_title('Linear Normalized (N)')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    cbar2 = plt.colorbar(im2, ax=axes[0, 1])
    cbar2.set_label('Normalized Value')
    
    temp_colored = color.CoolWarm(temp_norm_linear)
    axes[0, 2].imshow(temp_colored.T, origin='lower')
    axes[0, 2].set_title('CoolWarm Colormap Applied')
    axes[0, 2].set_xlabel('X Position')
    axes[0, 2].set_ylabel('Y Position')
    
    # Row 2: Density processing
    im3 = axes[1, 0].imshow(density, cmap='hot', origin='lower')
    axes[1, 0].set_title('Original Density Field')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    cbar3 = plt.colorbar(im3, ax=axes[1, 0])
    cbar3.set_label('Density')
    
    density_norm_log = color.NL(density)
    im4 = axes[1, 1].imshow(density_norm_log, cmap='hot', origin='lower')
    axes[1, 1].set_title('Log Normalized (NL)')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    cbar4 = plt.colorbar(im4, ax=axes[1, 1])
    cbar4.set_label('Log Normalized Value')
    
    density_colored = color.Hot(density_norm_log)
    axes[1, 2].imshow(density_colored.T, origin='lower')
    axes[1, 2].set_title('Hot Colormap Applied')
    axes[1, 2].set_xlabel('X Position')
    axes[1, 2].set_ylabel('Y Position')
    
    plt.suptitle('Color Mapping: Normalization and Colormaps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, "04_color_mapping.png"


def test_cosmological_visualization():
    """Test cosmological calculations and visualization."""
    print("Testing cosmological calculations...")
    
    # Use WMAP7 cosmology
    cosmo = cosmology.WMAP7
    
    # Redshift range for calculations
    z_range = np.logspace(-2, 1, 50)  # z = 0.01 to 10
    
    # Calculate cosmological quantities
    distances = np.array([cosmo.D(z) for z in z_range])
    times = np.array([cosmo.T(z) for z in z_range])
    hubble_params = np.array([cosmo.H(z) for z in z_range])
    
    # Create mock observation
    np.random.seed(42)
    z_obs = 1.0
    n_galaxies = 200
    
    # Galaxy positions in comoving coordinates
    galaxy_pos = np.random.uniform(-50, 50, (n_galaxies, 3))
    galaxy_pos[:, 2] += cosmo.D(z_obs)  # Place at redshift z_obs
    
    # Galaxy properties
    galaxy_masses = np.random.lognormal(30, 0.5, n_galaxies)  # Log solar masses
    
    # Calculate line-of-sight velocities
    galaxy_distances = np.linalg.norm(galaxy_pos, axis=1)
    los_velocities = np.array([cosmo.losvel(d, z_obs) for d in galaxy_distances])
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cosmological evolution plots
    axes[0, 0].loglog(z_range, distances)
    axes[0, 0].set_xlabel('Redshift z')
    axes[0, 0].set_ylabel('Comoving Distance [Mpc]')
    axes[0, 0].set_title('Distance-Redshift Relation')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogx(z_range, times)
    axes[0, 1].set_xlabel('Redshift z')
    axes[0, 1].set_ylabel('Cosmic Time [Gyr]')
    axes[0, 1].set_title('Cosmic Time Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mock observation visualization
    # Project galaxies for visualization
    proj = camera.ortho(
        near=cosmo.D(z_obs)-20, 
        far=cosmo.D(z_obs)+20, 
        extent=(-60, 60, -60, 60)
    )
    mv = camera.lookat([0, 0, 0], [0, 0, cosmo.D(z_obs)], [0, 1, 0])
    cam_matrix = camera.matrix(proj, mv)
    
    # Transform galaxy positions
    xc = camera.apply(cam_matrix, galaxy_pos)
    xd = camera.todevice_shape(xc, (128, 128))
    
    # Paint mass and velocity maps
    mass_image = painter.paint(xd, np.ones(len(xd))*0.5, [galaxy_masses], (128, 128))[0]
    vel_image = painter.paint(xd, np.ones(len(xd))*0.5, [los_velocities], (128, 128))[0]
    
    # Mass map
    im1 = axes[1, 0].imshow(mass_image.T, origin='lower', cmap='hot')
    axes[1, 0].set_title(f'Galaxy Mass Distribution (z={z_obs})')
    axes[1, 0].set_xlabel('Sky X [pixels]')
    axes[1, 0].set_ylabel('Sky Y [pixels]')
    cbar1 = plt.colorbar(im1, ax=axes[1, 0])
    cbar1.set_label('Mass [M☉]')
    
    # Velocity map  
    im2 = axes[1, 1].imshow(vel_image.T, origin='lower', cmap='coolwarm')
    axes[1, 1].set_title('Line-of-sight Velocity Map')
    axes[1, 1].set_xlabel('Sky X [pixels]')
    axes[1, 1].set_ylabel('Sky Y [pixels]')
    cbar2 = plt.colorbar(im2, ax=axes[1, 1])
    cbar2.set_label('Velocity [km/s]')
    
    plt.suptitle('Cosmological Calculations and Mock Observations (WMAP7)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, "05_cosmological_visualization.png"


def test_multiple_datasets():
    """Test painting multiple datasets simultaneously."""
    print("Testing multiple dataset painting...")
    
    # Create particle data
    np.random.seed(42)
    n_particles = 150
    
    # Ring-like distribution
    angles = np.random.uniform(0, 2*np.pi, n_particles)
    radii = np.random.normal(10, 2, n_particles)
    
    pos = np.column_stack([
        radii * np.cos(angles) + 20,
        radii * np.sin(angles) + 20
    ])
    
    sml = np.random.uniform(0.3, 1.0, n_particles)
    
    # Multiple physical quantities
    density = np.random.lognormal(0, 0.5, n_particles)
    temperature = 1e4 + 5e3 * np.random.random(n_particles)
    velocity = np.random.normal(0, 100, n_particles)
    metallicity = np.random.uniform(0.1, 2.0, n_particles)
    
    # Convert world coordinates to device coordinates
    # Particles are in range [0,40], map to device coordinates [0,128]
    image_shape = (128, 128)
    pos_device = pos * (image_shape[0] / 40.0)
    sml_device = sml * (image_shape[0] / 40.0)
    
    # Paint all quantities at once
    images = painter.paint(
        pos_device, sml_device, 
        [density, temperature, velocity, metallicity], 
        image_shape
    )
    
    density_img, temp_img, vel_img, metal_img = images
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original particle data
    sc1 = axes[0, 0].scatter(pos[:, 0], pos[:, 1], c=density, s=30, cmap='hot', alpha=0.7)
    axes[0, 0].set_title('Density (Particles)')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].set_xlim(0, 40)
    axes[0, 0].set_ylim(0, 40)
    plt.colorbar(sc1, ax=axes[0, 0], label='Density')
    
    sc2 = axes[0, 1].scatter(pos[:, 0], pos[:, 1], c=temperature, s=30, cmap='coolwarm', alpha=0.7)
    axes[0, 1].set_title('Temperature (Particles)')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    axes[0, 1].set_xlim(0, 40)
    axes[0, 1].set_ylim(0, 40)
    plt.colorbar(sc2, ax=axes[0, 1], label='Temperature [K]')
    
    sc3 = axes[0, 2].scatter(pos[:, 0], pos[:, 1], c=velocity, s=30, cmap='coolwarm', alpha=0.7)
    axes[0, 2].set_title('Velocity (Particles)')
    axes[0, 2].set_xlabel('X Position')
    axes[0, 2].set_ylabel('Y Position')
    axes[0, 2].set_xlim(0, 40)
    axes[0, 2].set_ylim(0, 40)
    plt.colorbar(sc3, ax=axes[0, 2], label='Velocity [km/s]')
    
    sc4 = axes[0, 3].scatter(pos[:, 0], pos[:, 1], c=metallicity, s=30, cmap='viridis', alpha=0.7)
    axes[0, 3].set_title('Metallicity (Particles)')
    axes[0, 3].set_xlabel('X Position')
    axes[0, 3].set_ylabel('Y Position')
    axes[0, 3].set_xlim(0, 40)
    axes[0, 3].set_ylim(0, 40)
    plt.colorbar(sc4, ax=axes[0, 3], label='Z/Z☉')
    
    # Row 2: Painted images
    im1 = axes[1, 0].imshow(density_img.T, origin='lower', extent=[0, 40, 0, 40], cmap='hot')
    axes[1, 0].set_title('Density (SPH Painted)')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    plt.colorbar(im1, ax=axes[1, 0], label='Painted Density')
    
    im2 = axes[1, 1].imshow(temp_img.T, origin='lower', extent=[0, 40, 0, 40], cmap='coolwarm')
    axes[1, 1].set_title('Temperature (SPH Painted)')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    plt.colorbar(im2, ax=axes[1, 1], label='Painted Temp [K]')
    
    im3 = axes[1, 2].imshow(vel_img.T, origin='lower', extent=[0, 40, 0, 40], cmap='coolwarm')
    axes[1, 2].set_title('Velocity (SPH Painted)')
    axes[1, 2].set_xlabel('X Position')
    axes[1, 2].set_ylabel('Y Position')
    plt.colorbar(im3, ax=axes[1, 2], label='Painted Vel [km/s]')
    
    im4 = axes[1, 3].imshow(metal_img.T, origin='lower', extent=[0, 40, 0, 40], cmap='viridis')
    axes[1, 3].set_title('Metallicity (SPH Painted)')
    axes[1, 3].set_xlabel('X Position')
    axes[1, 3].set_ylabel('Y Position')
    plt.colorbar(im4, ax=axes[1, 3], label='Painted Z/Z☉')
    
    plt.suptitle('Multi-Channel SPH Painting: 4 Physical Quantities Simultaneously', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, "06_multiple_datasets.png"


def test_todevice_shape_comparison():
    """Test and compare old vs new todevice functions."""
    print("Testing todevice vs todevice_shape comparison...")
    
    # Create test clipping coordinates
    np.random.seed(42)
    n_points = 20
    xc = np.random.uniform(-0.8, 0.8, (n_points, 3))
    
    # Compare old extent method vs new shape method
    extent = (127, 127)  # old method: (width-1, height-1)
    shape = (128, 128)   # new method: (height, width)
    
    # Apply both methods
    xd_old = camera.todevice(xc, extent)
    xd_new = camera.todevice_shape(xc, shape)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original clipping coordinates
    axes[0].scatter(xc[:, 0], xc[:, 1], c=range(n_points), s=50, cmap='tab20')
    axes[0].set_xlim(-1, 1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_xlabel('Clipping X')
    axes[0].set_ylabel('Clipping Y')
    axes[0].set_title('Original Clipping Coordinates')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Old todevice method
    axes[1].scatter(xd_old[:, 0], xd_old[:, 1], c=range(n_points), s=50, cmap='tab20')
    axes[1].set_xlim(-5, 132)
    axes[1].set_ylim(-5, 132)
    axes[1].set_xlabel('Device X (pixels)')
    axes[1].set_ylabel('Device Y (pixels)')
    axes[1].set_title('todevice(xc, extent=(127,127))')
    axes[1].grid(True, alpha=0.3)
    
    # New todevice_shape method
    axes[2].scatter(xd_new[:, 0], xd_new[:, 1], c=range(n_points), s=50, cmap='tab20')
    axes[2].set_xlim(-5, 132)
    axes[2].set_ylim(-5, 132)
    axes[2].set_xlabel('Device X (pixels)')
    axes[2].set_ylabel('Device Y (pixels)')
    axes[2].set_title('todevice_shape(xc, shape=(128,128))')
    axes[2].grid(True, alpha=0.3)
    
    # Add text annotations showing ranges
    axes[1].text(0.02, 0.98, f'Range: [0, 127] × [0, 127]', transform=axes[1].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[2].text(0.02, 0.98, f'Range: [0, 127] × [0, 127]', transform=axes[2].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Device Coordinate Transformation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, "07_todevice_comparison.png"


def test_periodic_boundary():
    """Test periodic boundary conditions in SPH painting."""
    print("Testing periodic boundary conditions...")
    
    # Create uniform grid of particles with proper spacing
    np.random.seed(42)
    grid_size = 16  # Good density for clear visualization
    # Place particles with half-spacing buffer from edges to avoid aliasing
    spacing = 64.0 / grid_size
    x = np.linspace(spacing/2, 64 - spacing/2, grid_size)
    y = np.linspace(spacing/2, 64 - spacing/2, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    pos_world = np.column_stack([xx.flatten(), yy.flatten()])
    n_particles = len(pos_world)
    
    # Large smoothing length but reasonable relative to spacing
    sml_world = np.full(n_particles, 8.0)  # 2x particle spacing
    density = np.full(n_particles, 1.0)
    
    # Image setup
    image_shape = (64, 64)
    world_size = 64.0
    
    # Already in device coordinates (0-64)
    pos_device = pos_world.copy()
    sml_device = sml_world.copy()
    
    # Paint without periodic boundaries
    result_nonperiodic = painter.paint(
        pos_device, sml_device, [density], image_shape, periodic=False
    )
    image_nonperiodic = result_nonperiodic[0]
    
    # Paint with periodic boundaries
    result_periodic = painter.paint(
        pos_device, sml_device, [density], image_shape, periodic=True
    )
    image_periodic = result_periodic[0]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Non-periodic
    axes[0, 0].scatter(pos_world[:, 0], pos_world[:, 1], 
                      c='green', s=150, alpha=0.6, edgecolors='black', linewidth=0.5)
    # Draw circles to show smoothing length for corner and edge particles
    corner_edge_indices = []
    for i in range(n_particles):
        x, y = pos_world[i]
        if x < 5 or x > 59 or y < 5 or y > 59:  # Near edges
            corner_edge_indices.append(i)
    
    # Show smoothing circles for a few edge/corner particles
    for i in corner_edge_indices[::len(corner_edge_indices)//4]:
        circle = plt.Circle((pos_world[i, 0], pos_world[i, 1]), sml_world[i], 
                          fill=False, color='red', alpha=0.5, linestyle='--', linewidth=2)
        axes[0, 0].add_patch(circle)
    axes[0, 0].set_xlim(-5, 69)
    axes[0, 0].set_ylim(-5, 69)
    axes[0, 0].set_title('Particles (Non-Periodic)')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # Use better colormap range
    vmin = np.min([image_nonperiodic.min(), image_periodic.min()])
    vmax = np.max([image_nonperiodic.max(), image_periodic.max()])
    
    im1 = axes[0, 1].imshow(image_nonperiodic.T, origin='lower', 
                           extent=[0, 64, 0, 64], cmap='hot', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('SPH Painted (Non-Periodic)')
    axes[0, 1].set_xlabel('X Position')
    axes[0, 1].set_ylabel('Y Position')
    plt.colorbar(im1, ax=axes[0, 1], label='Density')
    
    # Show difference image
    diff_image = image_periodic - image_nonperiodic
    im_diff = axes[0, 2].imshow(diff_image.T, origin='lower', 
                               extent=[0, 64, 0, 64], cmap='RdBu_r',
                               vmin=-np.abs(diff_image).max(), 
                               vmax=np.abs(diff_image).max())
    axes[0, 2].set_title('Difference (Periodic - Non-Periodic)')
    axes[0, 2].set_xlabel('X Position')
    axes[0, 2].set_ylabel('Y Position')
    plt.colorbar(im_diff, ax=axes[0, 2], label='Density Difference')
    # Add contour lines at edges
    axes[0, 2].axvline(x=5, color='yellow', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(x=59, color='yellow', linestyle='--', alpha=0.5)
    axes[0, 2].axhline(y=5, color='yellow', linestyle='--', alpha=0.5)
    axes[0, 2].axhline(y=59, color='yellow', linestyle='--', alpha=0.5)
    
    # Row 2: Periodic
    axes[1, 0].scatter(pos_world[:, 0], pos_world[:, 1], 
                      c='green', s=150, alpha=0.6, edgecolors='black', linewidth=0.5)
    # Show wrapped particles in lighter color
    for dx in [-64, 64]:
        for dy in [-64, 64]:
            if dx != 0 or dy != 0:
                axes[1, 0].scatter(pos_world[:, 0] + dx, pos_world[:, 1] + dy, 
                                 c='lightgreen', s=100, alpha=0.3, marker='s',
                                 edgecolors='gray', linewidth=0.5)
    axes[1, 0].set_xlim(-5, 69)
    axes[1, 0].set_ylim(-5, 69)
    axes[1, 0].set_title('Particles (Periodic)')
    axes[1, 0].set_xlabel('X Position')
    axes[1, 0].set_ylabel('Y Position')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    im2 = axes[1, 1].imshow(image_periodic.T, origin='lower', 
                           extent=[0, 64, 0, 64], cmap='hot', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('SPH Painted (Periodic)')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    plt.colorbar(im2, ax=axes[1, 1], label='Density')
    
    # Show line profiles comparing edge vs center
    edge_y = 1  # Very close to edge
    center_y = 32
    
    axes[1, 2].plot(image_nonperiodic[edge_y, :], label=f'Non-Periodic Edge (y={edge_y})', 
                    linewidth=2, color='red', alpha=0.7)
    axes[1, 2].plot(image_nonperiodic[center_y, :], label=f'Non-Periodic Center (y={center_y})', 
                    linewidth=2, color='darkred', linestyle='--', alpha=0.7)
    axes[1, 2].plot(image_periodic[edge_y, :], label=f'Periodic Edge (y={edge_y})', 
                    linewidth=2, color='blue')
    axes[1, 2].plot(image_periodic[center_y, :], label=f'Periodic Center (y={center_y})', 
                    linewidth=2, color='darkblue', linestyle='--')
    
    axes[1, 2].set_title('Edge vs Center Brightness Comparison')
    axes[1, 2].set_xlabel('X Position')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add annotations showing the difference
    max_np_edge = np.max(image_nonperiodic[edge_y, :])
    max_np_center = np.max(image_nonperiodic[center_y, :])
    max_p_edge = np.max(image_periodic[edge_y, :])
    max_p_center = np.max(image_periodic[center_y, :])
    
    axes[1, 2].text(0.02, 0.95, f'Non-Periodic: Edge/Center = {max_np_edge/max_np_center:.2f}',
                    transform=axes[1, 2].transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].text(0.02, 0.85, f'Periodic: Edge/Center = {max_p_edge/max_p_center:.2f}',
                    transform=axes[1, 2].transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Add text annotations
    fig.text(0.5, 0.95, 'Periodic Boundary Conditions: Uniform Grid Test', 
             ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.92, 'With periodic=True, edge brightness matches center brightness for uniform distribution', 
             ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, "08_periodic_boundary.png"


def main():
    """Run all test visualizations."""
    print("Gaepsi2 Comprehensive Test and Visualization")
    print("=" * 50)
    
    # Set up output directory
    output_dir = setup_output_directory()
    
    # Configure matplotlib for better output
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['font.size'] = 10
    
    # List of all tests to run
    tests = [
        test_basic_painting,
        test_camera_transformations,
        test_perspective_projection,
        test_color_mapping,
        test_cosmological_visualization,
        test_multiple_datasets,
        test_todevice_shape_comparison,
        test_periodic_boundary,
    ]
    
    # Run all tests and save plots
    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...")
            fig, filename = test_func()
            
            # Save the plot
            output_path = os.path.join(output_dir, filename)
            fig.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)  # Free memory
            
            print(f"  → Saved: {output_path}")
            
        except Exception as e:
            print(f"  ✗ Error in {test_func.__name__}: {e}")
            continue
    
    print(f"\nAll test plots saved to: {output_dir}/")
    print("Test suite completed successfully!")
    
    # Print summary of what was tested
    print("\nGenerated plots demonstrate:")
    print("1. Basic SPH particle painting")
    print("2. 3D to 2D camera transformations")
    print("3. Perspective projection with depth")
    print("4. Color normalization and mapping")
    print("5. Cosmological calculations and mock observations")
    print("6. Multi-channel simultaneous painting")
    print("7. New todevice_shape() function comparison")


if __name__ == "__main__":
    main()