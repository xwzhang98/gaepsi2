"""Tests for painter module functionality."""
import numpy as np
import pytest
from gaepsi2 import painter


class TestPaintFunction:
    """Test SPH painting functionality."""
    
    def test_paint_basic(self):
        """Test basic painting functionality."""
        # Simple 2x2 grid with one particle at center
        pos = np.array([[0.5, 0.5]])
        sml = np.array([0.5])
        data = [np.array([1.0])]
        shape = (2, 2)
        
        result = painter.paint(pos, sml, data, shape)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].shape == shape
        assert np.all(result[0] >= 0)  # Painted values should be non-negative
    
    def test_paint_multiple_particles(self):
        """Test painting with multiple particles."""
        # 3x3 grid with particles at corners
        pos = np.array([
            [0.0, 0.0],
            [1.0, 1.0], 
            [2.0, 2.0]
        ])
        sml = np.array([0.5, 0.5, 0.5])
        data = [np.array([1.0, 2.0, 3.0])]
        shape = (3, 3)
        
        result = painter.paint(pos, sml, data, shape)
        
        assert result[0].shape == shape
        assert np.sum(result[0]) > 0  # Should have painted something
    
    def test_paint_multiple_datasets(self):
        """Test painting multiple data arrays simultaneously."""
        pos = np.array([[1.0, 1.0]])
        sml = np.array([0.5])
        data = [
            np.array([1.0]),   # density
            np.array([2.0]),   # temperature  
            np.array([3.0])    # pressure
        ]
        shape = (3, 3)
        
        result = painter.paint(pos, sml, data, shape)
        
        assert len(result) == 3
        for i, res in enumerate(result):
            assert res.shape == shape
            assert np.sum(res) > 0
    
    def test_paint_with_mask(self):
        """Test painting with particle mask."""
        pos = np.array([
            [0.5, 0.5],
            [1.5, 1.5]
        ])
        sml = np.array([0.5, 0.5])
        data = [np.array([1.0, 2.0])]
        shape = (3, 3)
        mask = np.array([True, False])  # Only paint first particle
        
        result = painter.paint(pos, sml, data, shape, mask=mask)
        
        # Should only have contribution from first particle
        assert result[0].shape == shape
        # The exact values depend on kernel implementation
        assert np.sum(result[0]) > 0
    
    def test_paint_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Particle outside image bounds
        pos = np.array([[-1.0, -1.0]])
        sml = np.array([0.1])
        data = [np.array([1.0])]
        shape = (2, 2)
        
        result = painter.paint(pos, sml, data, shape)
        
        # Should handle gracefully (might be zero or small values)
        assert result[0].shape == shape
        assert np.all(np.isfinite(result[0]))
    
    def test_paint_parallel_processing(self):
        """Test painting with parallel processing."""
        # Large number of particles to test parallelization
        np.random.seed(42)
        n_particles = 100
        pos = np.random.uniform(0, 10, (n_particles, 2))
        sml = np.random.uniform(0.1, 0.5, n_particles)
        data = [np.random.uniform(0, 1, n_particles)]
        shape = (10, 10)
        
        # Test with different number of processes
        result_serial = painter.paint(pos, sml, data, shape, np=1)
        result_parallel = painter.paint(pos, sml, data, shape, np=2)
        
        # Results should be identical (or very close due to floating point)
        assert result_serial[0].shape == result_parallel[0].shape
        assert np.allclose(result_serial[0], result_parallel[0], atol=1e-12)
    
    def test_paint_zero_smoothing(self):
        """Test behavior with very small smoothing lengths."""
        pos = np.array([[1.0, 1.0]])
        sml = np.array([1e-10])  # Very small smoothing
        data = [np.array([1.0])]
        shape = (3, 3)
        
        result = painter.paint(pos, sml, data, shape)
        
        # Should handle without crashing
        assert result[0].shape == shape
        assert np.all(np.isfinite(result[0]))
    
    def test_paint_large_smoothing(self):
        """Test behavior with very large smoothing lengths."""
        pos = np.array([[1.0, 1.0]])
        sml = np.array([10.0])  # Large smoothing relative to image
        data = [np.array([1.0])]
        shape = (3, 3)
        
        result = painter.paint(pos, sml, data, shape)
        
        # Should distribute mass across many pixels
        assert result[0].shape == shape
        assert np.sum(result[0] > 0) > 1  # Multiple pixels should have values


class TestInputValidation:
    """Test input validation for painter functions."""
    
    def test_mismatched_array_sizes(self):
        """Test error handling for mismatched input arrays."""
        pos = np.array([[0, 0], [1, 1]])  # 2 particles
        sml = np.array([0.5])             # 1 smoothing length
        data = [np.array([1.0, 2.0])]     # 2 data values
        shape = (3, 3)
        
        # Should raise an error due to size mismatch
        with pytest.raises((ValueError, IndexError)):
            painter.paint(pos, sml, data, shape)
    
    def test_invalid_shape(self):
        """Test error handling for invalid shape."""
        pos = np.array([[0.5, 0.5]])
        sml = np.array([0.5])
        data = [np.array([1.0])]
        
        # Invalid shapes
        invalid_shapes = [
            (0, 2),    # Zero dimension
            (-1, 2),   # Negative dimension
            (2.5, 3),  # Non-integer dimension
        ]
        
        for shape in invalid_shapes:
            with pytest.raises((ValueError, TypeError)):
                painter.paint(pos, sml, data, shape)
    
    def test_empty_arrays(self):
        """Test behavior with empty input arrays."""
        pos = np.array([]).reshape(0, 2)
        sml = np.array([])
        data = [np.array([])]
        shape = (3, 3)
        
        result = painter.paint(pos, sml, data, shape)
        
        # Should return array of zeros
        assert result[0].shape == shape
        assert np.all(result[0] == 0)
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinite values."""
        pos = np.array([[1.0, 1.0], [np.nan, np.inf]])
        sml = np.array([0.5, 0.5])
        data = [np.array([1.0, 2.0])]
        shape = (3, 3)
        
        # Should handle gracefully without crashing
        try:
            result = painter.paint(pos, sml, data, shape)
            # Result should be finite where valid
            assert np.all(np.isfinite(result[0]) | (result[0] == 0))
        except (ValueError, RuntimeError):
            # Acceptable to raise error for invalid input
            pass


class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_dataset(self):
        """Test with larger dataset to check performance."""
        np.random.seed(42)
        n_particles = 1000
        pos = np.random.uniform(0, 50, (n_particles, 2))
        sml = np.random.uniform(0.1, 1.0, n_particles)
        data = [np.random.uniform(0, 1, n_particles)]
        shape = (50, 50)
        
        # Should complete in reasonable time
        result = painter.paint(pos, sml, data, shape)
        
        assert result[0].shape == shape
        assert np.sum(result[0]) > 0
    
    def test_memory_efficiency(self):
        """Test that function doesn't use excessive memory."""
        # This is more of a smoke test - actual memory testing
        # would require more sophisticated tooling
        np.random.seed(42)
        n_particles = 500
        pos = np.random.uniform(0, 20, (n_particles, 2))
        sml = np.random.uniform(0.1, 0.5, n_particles)
        
        # Multiple datasets
        data = [np.random.uniform(0, 1, n_particles) for _ in range(5)]
        shape = (20, 20)
        
        result = painter.paint(pos, sml, data, shape)
        
        assert len(result) == 5
        for res in result:
            assert res.shape == shape


if __name__ == "__main__":
    pytest.main([__file__])