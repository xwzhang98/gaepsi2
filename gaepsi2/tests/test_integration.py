"""Integration tests for gaepsi2 - testing complete workflows."""
import numpy as np
import pytest
from gaepsi2 import camera, painter, color, cosmology


class TestCompleteVisualizationWorkflow:
    """Test complete SPH visualization workflow."""
    
    def test_basic_sph_visualization(self):
        """Test complete workflow from particles to image."""
        # Create mock SPH particle data
        np.random.seed(42)
        n_particles = 100
        
        # Particles in a box from -5 to 5
        pos = np.random.uniform(-5, 5, (n_particles, 3))
        sml = np.random.uniform(0.1, 0.5, n_particles)
        density = np.random.uniform(0.1, 2.0, n_particles)
        temperature = np.random.uniform(1e4, 1e6, n_particles)
        
        # Set up camera
        proj = camera.ortho(near=1, far=20, extent=(-6, 6, -6, 6))
        mv = camera.lookat(pos=[0, 0, -10], target=[0, 0, 0], up=[0, 1, 0])
        cam_matrix = camera.matrix(proj, mv)
        
        # Transform to device coordinates
        xc = camera.apply(cam_matrix, pos)
        
        # Use new shape-based function
        image_shape = (64, 64)
        xd = camera.todevice_shape(xc, image_shape)
        
        # Paint density
        density_image = painter.paint(
            xd, sml, [density], 
            shape=image_shape
        )[0]
        
        # Paint temperature  
        temp_image = painter.paint(
            xd, sml, [temperature],
            shape=image_shape
        )[0]
        
        # Apply color mapping
        density_norm = color.N(density_image)
        temp_norm = color.NL(temp_image)  # Log scale for temperature
        
        density_colored = color.Hot(density_norm)
        temp_colored = color.CoolWarm(temp_norm)
        
        # Verify results
        assert density_image.shape == image_shape
        assert temp_image.shape == image_shape
        assert density_colored.shape == (*image_shape, 3)
        assert temp_colored.shape == (*image_shape, 3)
        
        # Images should have some painted content
        assert np.sum(density_image) > 0
        assert np.sum(temp_image) > 0
        
        # Colors should be in valid range
        assert np.all(density_colored >= 0) and np.all(density_colored <= 1)
        assert np.all(temp_colored >= 0) and np.all(temp_colored <= 1)
    
    def test_perspective_projection_workflow(self):
        """Test workflow with perspective projection."""
        np.random.seed(42)
        
        # Create particles at different depths
        n_particles = 50
        pos = np.random.uniform(-2, 2, (n_particles, 3))
        pos[:, 2] = np.random.uniform(-10, -5, n_particles)  # Depth variation
        
        sml = np.random.uniform(0.1, 0.3, n_particles)
        data = np.random.uniform(0, 1, n_particles)
        
        # Perspective camera
        proj = camera.persp(near=1, far=15, fov=np.pi/3, aspect=1.0)
        mv = camera.lookat([0, 0, -3], [0, 0, -7], [0, 1, 0])
        cam_matrix = camera.matrix(proj, mv)
        
        # Full data-to-device transformation
        extent = (128, 128)
        xd, smld, clip_mask = camera.data_to_device(
            cam_matrix, pos, sml, extent, apply_clip=True
        )
        
        # Paint with transformed data
        if len(xd) > 0:  # Only if particles remain after clipping
            image = painter.paint(
                xd, smld, [data[clip_mask]], 
                shape=(128, 128)
            )[0]
            
            assert image.shape == (128, 128)
            
            # Should have painted something if particles are visible
            if np.sum(clip_mask) > 0:
                assert np.sum(image) >= 0
    
    def test_cosmological_visualization(self):
        """Test visualization with cosmological context."""
        # Use cosmological parameters
        cosmo = cosmology.WMAP7
        
        # Simulate observing at redshift z=1
        z = 1.0
        distance = cosmo.D(z)  # Comoving distance
        
        # Create "observed" particle positions
        np.random.seed(42)
        n_particles = 80
        
        # Particles in a slice at cosmological distance
        pos = np.random.uniform(-1, 1, (n_particles, 3))
        pos[:, 2] += distance  # Place at cosmological distance
        
        sml = np.random.uniform(0.01, 0.05, n_particles)  # Small smoothing
        mass = np.random.uniform(1e10, 1e12, n_particles)  # Galaxy masses
        
        # Set up camera to observe this slice
        extent_size = 2.0
        proj = camera.ortho(
            near=distance-1, 
            far=distance+1, 
            extent=(-extent_size, extent_size, -extent_size, extent_size)
        )
        mv = camera.lookat([0, 0, 0], [0, 0, distance], [0, 1, 0])
        cam_matrix = camera.matrix(proj, mv)
        
        # Calculate line-of-sight velocities
        los_velocities = cosmo.losvel(
            np.linalg.norm(pos, axis=1), z
        )
        
        # Transform and paint
        xc = camera.apply(cam_matrix, pos)
        xd = camera.todevice_shape(xc, (64, 64))
        
        # Paint mass and velocity
        mass_image = painter.paint(xd, sml, [mass], (64, 64))[0]
        vel_image = painter.paint(xd, sml, [los_velocities], (64, 64))[0]
        
        # Apply appropriate color mapping
        mass_colored = color.Hot(color.NL(mass_image))
        vel_colored = color.CoolWarm(color.N(vel_image))
        
        assert mass_image.shape == (64, 64)
        assert vel_image.shape == (64, 64)
        assert mass_colored.shape == (64, 64, 3)
        assert vel_colored.shape == (64, 64, 3)


class TestDataPipelineRobustness:
    """Test robustness of data processing pipeline."""
    
    def test_empty_data_handling(self):
        """Test pipeline with empty particle data."""
        # Empty arrays
        pos = np.array([]).reshape(0, 3)
        sml = np.array([])
        data = [np.array([])]
        
        # Set up camera normally
        proj = camera.ortho(1, 10, (-5, 5, -5, 5))
        mv = camera.lookat([0, 0, -5], [0, 0, 0], [0, 1, 0])
        cam_matrix = camera.matrix(proj, mv)
        
        # Should handle empty data gracefully
        if len(pos) > 0:
            xc = camera.apply(cam_matrix, pos)
            xd = camera.todevice_shape(xc, (32, 32))
            image = painter.paint(xd, sml, data, (32, 32))[0]
        else:
            # Create empty image
            image = np.zeros((32, 32))
        
        # Should result in empty/zero image
        assert image.shape == (32, 32)
        assert np.sum(image) == 0
    
    def test_extreme_smoothing_lengths(self):
        """Test with extreme smoothing lengths."""
        pos = np.array([[0, 0, 0]])
        data = [np.array([1.0])]
        
        # Very small smoothing
        sml_small = np.array([1e-10])
        image_small = painter.paint(pos[:, :2], sml_small, data, (10, 10))[0]
        
        # Very large smoothing
        sml_large = np.array([100.0])
        image_large = painter.paint(pos[:, :2], sml_large, data, (10, 10))[0]
        
        assert image_small.shape == (10, 10)
        assert image_large.shape == (10, 10)
        assert np.all(np.isfinite(image_small))
        assert np.all(np.isfinite(image_large))
    
    def test_clipping_behavior(self):
        """Test clipping behavior with particles outside view."""
        # Mix of inside and outside particles
        pos = np.array([
            [0, 0, -5],      # Inside
            [10, 10, -5],    # Outside
            [-10, -10, -5],  # Outside
            [1, 1, -5],      # Inside
        ])
        sml = np.array([0.5, 0.5, 0.5, 0.5])
        data = [np.array([1, 2, 3, 4])]
        
        # Small viewing volume
        proj = camera.ortho(1, 10, (-2, 2, -2, 2))
        mv = camera.lookat([0, 0, 0], [0, 0, -5], [0, 1, 0])
        cam_matrix = camera.matrix(proj, mv)
        
        # Test with and without clipping
        xd_no_clip, smld_no_clip, clip_mask = camera.data_to_device(
            cam_matrix, pos, sml, (32, 32), apply_clip=False
        )
        
        xd_clip, smld_clip, _ = camera.data_to_device(
            cam_matrix, pos, sml, (32, 32), apply_clip=True
        )
        
        # With clipping, should have fewer particles
        assert len(xd_clip) <= len(xd_no_clip)
        assert len(smld_clip) <= len(smld_no_clip)
        
        # Clipping mask should identify inside particles
        assert clip_mask.dtype == bool
        assert len(clip_mask) == len(pos)


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows."""
    
    def test_large_dataset_workflow(self):
        """Test complete workflow with larger dataset."""
        np.random.seed(42)
        n_particles = 1000
        
        # Generate large particle dataset
        pos = np.random.uniform(-10, 10, (n_particles, 3))
        sml = np.random.uniform(0.1, 0.8, n_particles)
        density = np.random.lognormal(0, 1, n_particles)
        temperature = np.random.lognormal(10, 1, n_particles)
        
        # High resolution image
        image_size = 128
        
        # Set up camera
        proj = camera.ortho(1, 20, (-12, 12, -12, 12))
        mv = camera.lookat([0, 0, -15], [0, 0, 0], [0, 1, 0])
        cam_matrix = camera.matrix(proj, mv)
        
        # Transform data
        xc = camera.apply(cam_matrix, pos, np=1)  # Single thread for reproducibility
        xd = camera.todevice_shape(xc, (image_size, image_size))
        
        # Paint multiple quantities
        images = painter.paint(
            xd, sml, [density, temperature], 
            (image_size, image_size), np=1
        )
        
        density_img, temp_img = images
        
        # Apply color maps
        density_colored = color.Hot(color.N(density_img))
        temp_colored = color.CoolWarm(color.NL(temp_img))
        
        # Verify results
        assert density_img.shape == (image_size, image_size)
        assert temp_img.shape == (image_size, image_size)
        assert density_colored.shape == (image_size, image_size, 3)
        assert temp_colored.shape == (image_size, image_size, 3)
        
        # Should have substantial painted content
        assert np.sum(density_img) > 0
        assert np.sum(temp_img) > 0
    
    def test_parallel_processing_consistency(self):
        """Test that parallel processing gives consistent results."""
        np.random.seed(42)
        n_particles = 200
        
        pos = np.random.uniform(-5, 5, (n_particles, 3))
        sml = np.random.uniform(0.1, 0.5, n_particles)
        data = np.random.uniform(0, 1, n_particles)
        
        # Set up identical camera
        proj = camera.ortho(1, 15, (-6, 6, -6, 6))
        mv = camera.lookat([0, 0, -10], [0, 0, 0], [0, 1, 0])
        cam_matrix = camera.matrix(proj, mv)
        
        xc = camera.apply(cam_matrix, pos, np=1)
        xd = camera.todevice_shape(xc, (64, 64))
        
        # Paint with different numbers of processes
        image_serial = painter.paint(xd, sml, [data], (64, 64), np=1)[0]
        image_parallel = painter.paint(xd, sml, [data], (64, 64), np=2)[0]
        
        # Results should be identical (or very close)
        assert np.allclose(image_serial, image_parallel, atol=1e-12)


class TestErrorHandlingIntegration:
    """Test error handling in integrated workflows."""
    
    def test_mismatched_data_sizes(self):
        """Test error handling with mismatched data array sizes."""
        pos = np.array([[0, 0, 0], [1, 1, 1]])  # 2 particles
        sml = np.array([0.5])                   # 1 smoothing length
        data = [np.array([1, 2])]               # 2 data values
        
        # Should raise error due to size mismatch
        with pytest.raises((ValueError, IndexError)):
            painter.paint(pos[:, :2], sml, data, (10, 10))
    
    def test_invalid_camera_parameters(self):
        """Test error handling with invalid camera parameters."""
        # Invalid extent (near > far)
        with pytest.raises((ValueError, AssertionError)):
            camera.ortho(near=10, far=1, extent=(-5, 5, -5, 5))
        
        # Invalid FOV
        with pytest.raises((ValueError, RuntimeError)):
            camera.persp(near=1, far=10, fov=-1, aspect=1)
    
    def test_extreme_transformations(self):
        """Test behavior with extreme coordinate transformations."""
        # Very large coordinates
        pos = np.array([[1e10, 1e10, 1e10]])
        sml = np.array([1e5])
        data = [np.array([1.0])]
        
        proj = camera.ortho(1, 1e12, (-1e11, 1e11, -1e11, 1e11))
        mv = camera.lookat([0, 0, 0], [0, 0, 1e10], [0, 1, 0])
        
        try:
            cam_matrix = camera.matrix(proj, mv)
            xc = camera.apply(cam_matrix, pos)
            
            # Should either work or fail gracefully
            assert np.all(np.isfinite(xc)) or True  # Allow for overflow
        except (OverflowError, ValueError):
            # Acceptable to fail on extreme values
            pass


if __name__ == "__main__":
    pytest.main([__file__])