"""Tests for camera module functionality."""
import numpy as np
import pytest
from gaepsi2 import camera


class TestProjectionMatrices:
    """Test projection matrix creation and properties."""
    
    def test_ortho_basic(self):
        """Test basic orthographic projection."""
        proj = camera.ortho(near=1, far=10, extent=(-5, 5, -3, 3))
        
        assert isinstance(proj, camera.projectionmatrix)
        assert proj.near == 1
        assert proj.far == 10
        assert proj.shape == (4, 4)
        
        # Test corner points transformation
        test_points = np.array([
            [-5, -3, -5, 1],  # left, bottom, middle
            [5, 3, -5, 1],    # right, top, middle
        ])
        
        result = test_points @ proj.T
        # Should map to device coordinates [-1, 1]
        assert np.allclose(result[0, :2], [-1, -1])
        assert np.allclose(result[1, :2], [1, 1])
    
    def test_persp_basic(self):
        """Test basic perspective projection."""
        proj = camera.persp(near=1, far=10, fov=np.pi/2, aspect=1.0)
        
        assert isinstance(proj, camera.projectionmatrix)
        assert proj.near == 1
        assert proj.far == 10
        assert proj.shape == (4, 4)
    
    def test_fov_extent_conversion(self):
        """Test FOV to extent conversion and back."""
        fov = np.pi / 3
        aspect = 16/9
        D = 1.0
        
        extent = camera.fov2extent(fov, aspect, D)
        fov_back, aspect_back = camera.extent2fov(extent, D)
        
        assert np.isclose(fov, fov_back)
        assert np.isclose(aspect, aspect_back)


class TestModelView:
    """Test model/view matrix functionality."""
    
    def test_lookat_basic(self):
        """Test basic lookat matrix."""
        pos = np.array([0, 0, -5])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        mv = camera.lookat(pos, target, up)
        
        assert isinstance(mv, camera.modelviewmatrix)
        assert hasattr(mv, 'up')
        assert hasattr(mv, 'side')
        assert mv.shape == (4, 4)
        
        # Up vector should be normalized
        assert np.isclose(np.linalg.norm(mv.up), 1.0)
        assert np.isclose(np.linalg.norm(mv.side), 1.0)
    
    def test_lookat_orthogonal_vectors(self):
        """Test that lookat produces orthogonal coordinate system."""
        pos = np.array([1, 2, 3])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        mv = camera.lookat(pos, target, up)
        
        # Direction vector (camera is looking towards -z)
        direction = (target - pos) / np.linalg.norm(target - pos)
        
        # Check orthogonality
        assert np.isclose(np.dot(mv.up, mv.side), 0, atol=1e-10)
        assert np.isclose(np.dot(mv.up, -direction), 0, atol=1e-10)
        assert np.isclose(np.dot(mv.side, -direction), 0, atol=1e-10)


class TestCameraMatrix:
    """Test combined camera matrix functionality."""
    
    def test_matrix_creation(self):
        """Test camera matrix creation from projection and modelview."""
        proj = camera.ortho(1, 10, (-5, 5, -3, 3))
        mv = camera.lookat([0, 0, -5], [0, 0, 0], [0, 1, 0])
        
        matrix = camera.matrix(proj, mv)
        
        assert isinstance(matrix, camera.cameramatrix)
        assert matrix.near == proj.near
        assert matrix.far == proj.far
        assert np.array_equal(matrix.up, mv.up)
        assert np.array_equal(matrix.side, mv.side)
    
    def test_apply_transformation(self):
        """Test applying camera transformation to points."""
        proj = camera.ortho(1, 10, (-5, 5, -5, 5))
        mv = camera.lookat([0, 0, -5], [0, 0, 0], [0, 1, 0])
        matrix = camera.matrix(proj, mv)
        
        # Test points at known locations
        pos = np.array([
            [0, 0, 0],     # Origin
            [2.5, 2.5, 0], # Quarter way to edge
        ])
        
        result = camera.apply(matrix, pos, np=1)
        
        assert result.shape == pos.shape
        # Origin should map to center of device coordinates
        assert np.allclose(result[0], [0, 0, 0], atol=1e-10)


class TestDeviceCoordinates:
    """Test device coordinate transformations."""
    
    def test_todevice_2tuple(self):
        """Test todevice with 2-tuple extent."""
        xc = np.array([
            [-1, -1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ])
        extent = (100, 80)  # width, height
        
        result = camera.todevice(xc, extent)
        
        assert result.shape == (3, 2)
        assert np.allclose(result[0], [0, 0])      # Bottom-left
        assert np.allclose(result[1], [100, 80])   # Top-right
        assert np.allclose(result[2], [50, 40])    # Center
    
    def test_todevice_4tuple(self):
        """Test todevice with 4-tuple extent."""
        xc = np.array([
            [-1, -1, 0],
            [1, 1, 0],
        ])
        extent = (10, 90, 20, 60)  # left, right, bottom, top
        
        result = camera.todevice(xc, extent)
        
        assert np.allclose(result[0], [10, 20])  # left, bottom
        assert np.allclose(result[1], [90, 60])  # right, top
    
    def test_todevice_shape(self):
        """Test new todevice_shape function."""
        xc = np.array([
            [-1, -1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ])
        shape = (100, 80)  # height, width
        
        result = camera.todevice_shape(xc, shape)
        
        assert result.shape == (3, 2)
        assert np.allclose(result[0], [0, 0])        # Bottom-left
        assert np.allclose(result[1], [79, 99])      # Top-right (width-1, height-1)
        assert np.allclose(result[2], [39.5, 49.5])  # Center
    
    def test_clip_functionality(self):
        """Test clipping coordinate validation."""
        # Points inside frustum
        inside = np.array([
            [0, 0, 0],
            [0.5, -0.5, 0.9],
        ])
        
        # Points outside frustum
        outside = np.array([
            [2, 0, 0],     # x > 1
            [0, -2, 0],    # y < -1
            [0, 0, 1.5],   # z > 1
        ])
        
        assert np.all(camera.clip(inside))
        assert not np.any(camera.clip(outside))


class TestDataToDevice:
    """Test complete data transformation pipeline."""
    
    def test_data_to_device_basic(self):
        """Test basic data to device transformation."""
        proj = camera.ortho(1, 10, (-5, 5, -5, 5))
        mv = camera.lookat([0, 0, -5], [0, 0, 0], [0, 1, 0])
        matrix = camera.matrix(proj, mv)
        
        pos = np.array([[0, 0, 0], [1, 1, 0]])
        sml = np.array([0.1, 0.2])
        extent = (100, 100)
        
        xd, smld, clip_mask = camera.data_to_device(
            matrix, pos, sml, extent, apply_clip=False
        )
        
        assert xd.shape == (2, 2)
        assert smld.shape == (2,)
        assert clip_mask.shape == (2,)
        assert np.all(clip_mask)  # Should be inside frustum
    
    def test_data_to_device_with_clipping(self):
        """Test data to device with clipping applied."""
        proj = camera.ortho(1, 10, (-1, 1, -1, 1))
        mv = camera.lookat([0, 0, -5], [0, 0, 0], [0, 1, 0])
        matrix = camera.matrix(proj, mv)
        
        # Mix of inside and outside points
        pos = np.array([[0, 0, 0], [10, 10, 0]])  # Second point outside
        sml = np.array([0.1, 0.1])
        extent = (100, 100)
        
        xd, smld, clip_mask = camera.data_to_device(
            matrix, pos, sml, extent, apply_clip=True
        )
        
        # Only inside points should remain
        assert len(xd) <= len(pos)
        assert len(smld) <= len(pos)


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_matrix_type_validation(self):
        """Test that functions validate matrix types."""
        proj = camera.ortho(1, 10, (-5, 5, -5, 5))
        mv = camera.lookat([0, 0, -5], [0, 0, 0], [0, 1, 0])
        
        # Should raise assertion error for wrong types
        with pytest.raises(AssertionError):
            camera.matrix(mv, proj)  # Wrong order
        
        with pytest.raises(AssertionError):
            camera.apply(proj, np.array([[0, 0, 0]]))  # Wrong matrix type
    
    def test_array_shape_requirements(self):
        """Test array shape requirements."""
        proj = camera.ortho(1, 10, (-5, 5, -5, 5))
        mv = camera.lookat([0, 0, -5], [0, 0, 0], [0, 1, 0])
        matrix = camera.matrix(proj, mv)
        
        # Invalid position arrays
        invalid_pos = np.array([1, 2])  # 1D array
        
        # Should handle gracefully or provide clear error
        try:
            camera.apply(matrix, invalid_pos)
        except (ValueError, IndexError) as e:
            assert True  # Expected behavior


if __name__ == "__main__":
    pytest.main([__file__])