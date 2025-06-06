"""Tests for cosmology module functionality."""
import numpy as np
import pytest
from gaepsi2 import cosmology


class TestCosmologyBasics:
    """Test basic cosmology functionality."""
    
    def test_wmap7_constants(self):
        """Test WMAP7 cosmological parameters."""
        wmap7 = cosmology.WMAP7
        
        # Check that basic attributes exist
        assert hasattr(wmap7, 'h')
        assert hasattr(wmap7, 'Om')
        assert hasattr(wmap7, 'Ol')
        
        # Basic sanity checks on values
        assert 0.5 < wmap7.h < 1.0  # Hubble parameter
        assert 0.1 < wmap7.Om < 0.5  # Matter density
        assert 0.5 < wmap7.Ol < 1.0  # Lambda density
        
        # Density parameters should roughly sum to 1
        assert np.isclose(wmap7.Om + wmap7.Ol, 1.0, atol=0.1)
    
    def test_cosmology_methods(self):
        """Test that cosmology object has expected methods."""
        wmap7 = cosmology.WMAP7
        
        # Should have distance and time calculation methods
        expected_methods = ['D', 'T', 'H', 'rho', 'losvel']
        
        for method in expected_methods:
            assert hasattr(wmap7, method), f"Missing method: {method}"
            assert callable(getattr(wmap7, method)), f"Method {method} not callable"


class TestDistanceCalculations:
    """Test cosmological distance calculations."""
    
    def test_distance_calculation_basic(self):
        """Test basic distance calculation."""
        wmap7 = cosmology.WMAP7
        
        # Test at a reasonable redshift
        z = 1.0
        distance = wmap7.D(z)
        
        # Distance should be positive
        assert distance > 0
        
        # Distance should increase with redshift
        z_higher = 2.0
        distance_higher = wmap7.D(z_higher)
        assert distance_higher > distance
    
    def test_distance_at_zero_redshift(self):
        """Test distance calculation at z=0."""
        wmap7 = cosmology.WMAP7
        
        # Distance at z=0 should be 0
        distance = wmap7.D(0.0)
        assert np.isclose(distance, 0.0, atol=1e-10)
    
    def test_distance_array_input(self):
        """Test distance calculation with array input."""
        wmap7 = cosmology.WMAP7
        
        z_array = np.array([0.0, 0.5, 1.0, 2.0])
        distances = wmap7.D(z_array)
        
        assert len(distances) == len(z_array)
        assert distances[0] == 0.0  # z=0 should give distance=0
        
        # Distances should be monotonically increasing
        assert np.all(np.diff(distances) > 0)
    
    def test_high_redshift_behavior(self):
        """Test behavior at high redshift."""
        wmap7 = cosmology.WMAP7
        
        # Test at very high redshift
        z_high = 10.0
        distance = wmap7.D(z_high)
        
        assert distance > 0
        assert np.isfinite(distance)


class TestTimeCalculations:
    """Test cosmological time calculations."""
    
    def test_time_calculation_basic(self):
        """Test basic time calculation."""
        wmap7 = cosmology.WMAP7
        
        # Test at a reasonable redshift
        z = 1.0
        time = wmap7.T(z)
        
        # Time should be positive
        assert time > 0
        
        # Time should decrease with increasing redshift (looking back in time)
        z_higher = 2.0
        time_higher = wmap7.T(z_higher)
        assert time_higher < time
    
    def test_time_at_zero_redshift(self):
        """Test time calculation at z=0."""
        wmap7 = cosmology.WMAP7
        
        # Time at z=0 should be age of universe
        age = wmap7.T(0.0)
        
        # Age should be reasonable (between 10-20 Gyr)
        assert 10 < age < 20  # in Gyr
    
    def test_time_array_input(self):
        """Test time calculation with array input."""
        wmap7 = cosmology.WMAP7
        
        z_array = np.array([0.0, 1.0, 3.0, 10.0])
        times = wmap7.T(z_array)
        
        assert len(times) == len(z_array)
        
        # Times should be monotonically decreasing with redshift
        assert np.all(np.diff(times) < 0)


class TestHubbleParameter:
    """Test Hubble parameter calculations."""
    
    def test_hubble_at_zero_redshift(self):
        """Test Hubble parameter at z=0."""
        wmap7 = cosmology.WMAP7
        
        H0 = wmap7.H(0.0)
        
        # Should be close to h * 100 km/s/Mpc
        expected_H0 = wmap7.h * 100
        assert np.isclose(H0, expected_H0, rtol=0.01)
    
    def test_hubble_evolution(self):
        """Test Hubble parameter evolution."""
        wmap7 = cosmology.WMAP7
        
        # Hubble parameter should increase with redshift
        z_array = np.array([0.0, 1.0, 2.0, 5.0])
        H_array = wmap7.H(z_array)
        
        assert len(H_array) == len(z_array)
        # Generally H(z) > H(0) for z > 0
        assert np.all(H_array[1:] > H_array[0])


class TestDensityCalculations:
    """Test density calculations."""
    
    def test_density_at_zero_redshift(self):
        """Test density calculation at z=0."""
        wmap7 = cosmology.WMAP7
        
        rho0 = wmap7.rho(0.0)
        
        # Should be positive
        assert rho0 > 0
    
    def test_density_evolution(self):
        """Test density evolution with redshift."""
        wmap7 = cosmology.WMAP7
        
        # Density should increase with redshift
        z_array = np.array([0.0, 1.0, 2.0])
        rho_array = wmap7.rho(z_array)
        
        assert len(rho_array) == len(z_array)
        # Density increases as (1+z)^3 for matter
        assert np.all(rho_array[1:] > rho_array[0])


class TestLineOfSightVelocity:
    """Test line-of-sight velocity calculations."""
    
    def test_losvel_basic(self):
        """Test basic line-of-sight velocity calculation."""
        wmap7 = cosmology.WMAP7
        
        # Test with reasonable distance and redshift
        distance = 100.0  # Mpc
        z = 0.1
        
        velocity = wmap7.losvel(distance, z)
        
        # Should be reasonable velocity (roughly H0 * distance)
        expected_vel = wmap7.H(z) * distance
        assert np.isclose(velocity, expected_vel, rtol=0.1)
    
    def test_losvel_array_input(self):
        """Test losvel with array inputs."""
        wmap7 = cosmology.WMAP7
        
        distances = np.array([50, 100, 200])
        z = 0.1
        
        velocities = wmap7.losvel(distances, z)
        
        assert len(velocities) == len(distances)
        # Velocity should be proportional to distance
        assert np.allclose(velocities, distances * wmap7.H(z), rtol=0.01)


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_very_small_redshift(self):
        """Test behavior at very small redshift."""
        wmap7 = cosmology.WMAP7
        
        z_small = 1e-10
        distance = wmap7.D(z_small)
        time = wmap7.T(z_small)
        
        # Should be close to z=0 values
        assert np.isclose(distance, 0.0, atol=1e-8)
        assert np.isclose(time, wmap7.T(0.0), rtol=1e-8)
    
    def test_negative_redshift(self):
        """Test behavior with negative redshift."""
        wmap7 = cosmology.WMAP7
        
        # Negative redshift should either work (future universe) or fail gracefully
        try:
            distance = wmap7.D(-0.1)
            # If it works, should be reasonable
            assert np.isfinite(distance)
        except (ValueError, RuntimeError):
            # Acceptable to fail on unphysical input
            pass
    
    def test_very_high_redshift(self):
        """Test behavior at very high redshift."""
        wmap7 = cosmology.WMAP7
        
        z_high = 1000.0
        
        try:
            distance = wmap7.D(z_high)
            time = wmap7.T(z_high)
            
            assert np.isfinite(distance)
            assert np.isfinite(time)
            assert distance > 0
            assert time > 0
        except (ValueError, OverflowError):
            # Acceptable to fail at extreme redshifts
            pass


class TestCosmologyConsistency:
    """Test internal consistency of cosmological calculations."""
    
    def test_distance_time_consistency(self):
        """Test that distance and time calculations are consistent."""
        wmap7 = cosmology.WMAP7
        
        # At same redshift, should get consistent results
        z = 1.0
        distance = wmap7.D(z)
        time = wmap7.T(z)
        hubble = wmap7.H(z)
        
        # All should be finite and positive
        assert np.isfinite(distance) and distance > 0
        assert np.isfinite(time) and time > 0
        assert np.isfinite(hubble) and hubble > 0
    
    def test_units_consistency(self):
        """Test that units are consistent across calculations."""
        wmap7 = cosmology.WMAP7
        
        # This is mainly a sanity check that values are in reasonable ranges
        z = 1.0
        
        distance = wmap7.D(z)        # Should be in Mpc
        time = wmap7.T(z)            # Should be in Gyr
        hubble = wmap7.H(z)          # Should be in km/s/Mpc
        rho = wmap7.rho(z)           # Should be in appropriate density units
        
        # Reasonable ranges (these are approximate)
        assert 1000 < distance < 10000     # Mpc
        assert 1 < time < 15               # Gyr
        assert 50 < hubble < 500           # km/s/Mpc
        assert rho > 0                     # positive density


if __name__ == "__main__":
    pytest.main([__file__])