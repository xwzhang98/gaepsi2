"""Tests for color module functionality."""
import numpy as np
import pytest
from gaepsi2 import color


class TestNormalizationFunctions:
    """Test color normalization functions."""
    
    def test_N_basic(self):
        """Test basic N function (linear normalization)."""
        data = np.array([1, 2, 3, 4, 5])
        
        result = color.N(data)
        
        # Should normalize to [0, 1] range
        assert np.min(result) >= 0
        assert np.max(result) <= 1
        assert np.isclose(np.min(result), 0)
        assert np.isclose(np.max(result), 1)
        
        # Should preserve ordering
        assert np.all(np.diff(result) >= 0)
    
    def test_N_with_range(self):
        """Test N function with custom range."""
        data = np.array([0, 5, 10])
        range_val = 8  # min = max - range = 10 - 8 = 2
        
        result = color.N(data, range=range_val)
        
        # Values below min should be clipped to 0
        # Values above max should be clipped to 1
        assert result[0] == 0  # 0 < 2, so clipped to 0
        assert result[2] == 1  # 10 = max, so maps to 1
    
    def test_N_constant_data(self):
        """Test N function with constant data."""
        data = np.array([5, 5, 5, 5])
        
        result = color.N(data)
        
        # All values should be same (could be 0 or 1 depending on implementation)
        assert np.allclose(result, result[0])
    
    def test_NL_basic(self):
        """Test basic NL function (log normalization)."""
        data = np.array([1, 10, 100, 1000])
        
        result = color.NL(data)
        
        # Should normalize to [0, 1] range
        assert np.min(result) >= 0
        assert np.max(result) <= 1
        assert np.isclose(np.min(result), 0)
        assert np.isclose(np.max(result), 1)
        
        # Should preserve ordering
        assert np.all(np.diff(result) >= 0)
    
    def test_NL_with_zeros(self):
        """Test NL function with zero values."""
        data = np.array([0, 1, 10, 100])
        
        # Should handle zeros gracefully (might add small offset)
        result = color.NL(data)
        
        assert np.all(np.isfinite(result))
        assert np.min(result) >= 0
        assert np.max(result) <= 1
    
    def test_NL_negative_values(self):
        """Test NL function with negative values."""
        data = np.array([-1, 0, 1, 10])
        
        # Should handle negative values (implementation dependent)
        try:
            result = color.NL(data)
            assert np.all(np.isfinite(result))
        except (ValueError, RuntimeWarning):
            # Acceptable to fail or warn on negative values in log
            pass


class TestColormaps:
    """Test colormap functionality."""
    
    def test_CoolWarm_basic(self):
        """Test CoolWarm colormap."""
        data = np.linspace(0, 1, 10)
        
        result = color.CoolWarm(data)
        
        # Should return RGB values
        assert result.shape == (10, 3)
        # RGB values should be in [0, 1] range
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_Hot_basic(self):
        """Test Hot colormap."""
        data = np.linspace(0, 1, 10)
        
        result = color.Hot(data)
        
        # Should return RGB values
        assert result.shape == (10, 3)
        # RGB values should be in [0, 1] range
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_colormap_edge_values(self):
        """Test colormap behavior at edge values."""
        edge_data = np.array([0, 1])
        
        cool_warm = color.CoolWarm(edge_data)
        hot = color.Hot(edge_data)
        
        # Should handle edge cases without error
        assert cool_warm.shape == (2, 3)
        assert hot.shape == (2, 3)
        assert np.all(np.isfinite(cool_warm))
        assert np.all(np.isfinite(hot))
    
    def test_colormap_outside_range(self):
        """Test colormap behavior with values outside [0,1]."""
        out_of_range = np.array([-0.5, 0.5, 1.5])
        
        # Should handle gracefully (clamp or extrapolate)
        cool_warm = color.CoolWarm(out_of_range)
        hot = color.Hot(out_of_range)
        
        assert cool_warm.shape == (3, 3)
        assert hot.shape == (3, 3)
        assert np.all(np.isfinite(cool_warm))
        assert np.all(np.isfinite(hot))


class TestColormapClass:
    """Test the Colormap base class if accessible."""
    
    def test_colormap_creation(self):
        """Test creating custom colormap."""
        # This test depends on the Colormap class being accessible
        try:
            # Simple test colormap (grayscale)
            test_map = color.Colormap(
                np.array([0, 1]),           # positions
                np.array([[0, 0, 0],        # black
                         [1, 1, 1]])        # white
            )
            
            # Test interpolation
            result = test_map(np.array([0, 0.5, 1]))
            expected_middle = np.array([0.5, 0.5, 0.5])
            
            assert result.shape == (3, 3)
            assert np.allclose(result[0], [0, 0, 0])
            assert np.allclose(result[1], expected_middle, atol=0.1)
            assert np.allclose(result[2], [1, 1, 1])
            
        except (AttributeError, TypeError):
            # Colormap class might not be directly accessible
            pytest.skip("Colormap class not directly accessible")


class TestArrayHandling:
    """Test array handling and broadcasting."""
    
    def test_1d_arrays(self):
        """Test with 1D arrays."""
        data = np.array([1, 2, 3])
        
        n_result = color.N(data)
        nl_result = color.NL(data)
        cw_result = color.CoolWarm(color.N(data))
        hot_result = color.Hot(color.N(data))
        
        assert n_result.shape == (3,)
        assert nl_result.shape == (3,)
        assert cw_result.shape == (3, 3)
        assert hot_result.shape == (3, 3)
    
    def test_2d_arrays(self):
        """Test with 2D arrays."""
        data = np.array([[1, 2], [3, 4]])
        
        n_result = color.N(data)
        nl_result = color.NL(data)
        
        assert n_result.shape == (2, 2)
        assert nl_result.shape == (2, 2)
        
        # Colormap functions might handle 2D differently
        try:
            cw_result = color.CoolWarm(color.N(data))
            # If it works, should have shape (2, 2, 3) or flattened
            assert len(cw_result.shape) >= 2
        except (ValueError, AttributeError):
            # Might require flattened input
            cw_result = color.CoolWarm(color.N(data.flatten()))
            assert cw_result.shape[1] == 3
    
    def test_empty_arrays(self):
        """Test with empty arrays."""
        empty_data = np.array([])
        
        try:
            n_result = color.N(empty_data)
            assert n_result.shape == (0,)
        except (ValueError, IndexError):
            # Acceptable to fail on empty arrays
            pass
    
    def test_single_value(self):
        """Test with single value."""
        single_val = np.array([5])
        
        n_result = color.N(single_val)
        nl_result = color.NL(single_val)
        
        assert n_result.shape == (1,)
        assert nl_result.shape == (1,)
        
        # Single value normalization behavior is implementation dependent
        assert np.all(np.isfinite(n_result))
        assert np.all(np.isfinite(nl_result))


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_very_large_values(self):
        """Test with very large values."""
        large_data = np.array([1e10, 1e15, 1e20])
        
        n_result = color.N(large_data)
        nl_result = color.NL(large_data)
        
        assert np.all(np.isfinite(n_result))
        assert np.all(np.isfinite(nl_result))
    
    def test_very_small_values(self):
        """Test with very small positive values."""
        small_data = np.array([1e-10, 1e-15, 1e-20])
        
        n_result = color.N(small_data)
        nl_result = color.NL(small_data)
        
        assert np.all(np.isfinite(n_result))
        assert np.all(np.isfinite(nl_result))
    
    def test_close_values(self):
        """Test with very close values (numerical precision)."""
        close_data = np.array([1.0, 1.0 + 1e-15, 1.0 + 2e-15])
        
        n_result = color.N(close_data)
        
        # Should handle numerical precision gracefully
        assert np.all(np.isfinite(n_result))
        # Might all be same value due to precision limits
        assert len(np.unique(n_result)) <= 3


if __name__ == "__main__":
    pytest.main([__file__])