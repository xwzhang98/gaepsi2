"""
Legacy test file - basic smoke tests for all modules.

For comprehensive testing, run the individual test modules:
- test_camera.py
- test_painter.py  
- test_color.py
- test_cosmology.py
- test_integration.py
"""
import numpy
import pytest


def test_svr():
    """Test SVR module basic functionality."""
    from gaepsi2 import svr
    M = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    result = svr.remap([[0, 0, 1.]], M)
    assert result is not None
    print("SVR test passed:", result)


def test_painter():
    """Test painter module basic functionality.""" 
    from gaepsi2 import painter
    pos = numpy.array([
        numpy.arange(0, 1),
        numpy.arange(0, 1),
    ]).T
    sml = numpy.ones(len(pos)) * 4
    data = numpy.ones(len(pos))
    
    result1 = painter.paint(pos, sml, [data], (5, 5))
    result2 = painter.paint(pos + 0.5, sml, [data], (5, 5))
    
    assert len(result1) == 1
    assert len(result2) == 1 
    assert result1[0].shape == (5, 5)
    assert result2[0].shape == (5, 5)
    print("Painter test passed")


def test_camera():
    """Test camera module basic functionality."""
    from gaepsi2 import camera

    pos = numpy.random.uniform(size=(1000, 3)) * 20.
    pos -= 10.
    proj = camera.ortho(0, 20, (-10, 10, -10, 10))
    mv = camera.lookat((0, 0, -10), (0, 0, 0), (0, 1, 0))
    matrix = camera.matrix(proj, mv)
    p2d = camera.apply(matrix, pos)
    
    assert p2d.shape == pos.shape
    print("Camera test passed")


def test_color():
    """Test color module basic functionality."""
    from gaepsi2 import color
    
    data = numpy.array([1, 2, 3, 4, 5])
    
    # Test normalization functions
    n_result = color.N(data)
    nl_result = color.NL(data)
    
    # Test colormaps
    cw_result = color.CoolWarm(n_result)
    hot_result = color.Hot(n_result)
    
    assert n_result.shape == data.shape
    assert nl_result.shape == data.shape
    assert cw_result.shape == (*data.shape, 3)
    assert hot_result.shape == (*data.shape, 3)
    print("Color test passed")


def test_cosmology():
    """Test cosmology module basic functionality."""
    from gaepsi2 import cosmology
    
    wmap7 = cosmology.WMAP7
    
    # Test basic calculations
    distance = wmap7.D(1.0)
    time = wmap7.T(1.0)
    hubble = wmap7.H(1.0)
    rho = wmap7.rho(1.0)
    
    assert distance > 0
    assert time > 0
    assert hubble > 0
    assert rho > 0
    print("Cosmology test passed")


def test_todevice_shape():
    """Test new todevice_shape function."""
    from gaepsi2 import camera
    
    # Test clipping coordinates to device with shape
    xc = numpy.array([
        [-1, -1, 0],
        [1, 1, 0], 
        [0, 0, 0],
    ])
    shape = (100, 80)  # height, width
    
    result = camera.todevice_shape(xc, shape)
    
    assert result.shape == (3, 2)
    # Check that coordinates are in expected range [0, width-1], [0, height-1]
    assert numpy.allclose(result[0], [0, 0])      # Bottom-left
    assert numpy.allclose(result[1], [79, 99])    # Top-right  
    assert numpy.allclose(result[2], [39.5, 49.5]) # Center
    print("todevice_shape test passed")


if __name__ == "__main__":
    # Run all tests
    test_svr()
    test_painter()
    test_camera()
    test_color()
    test_cosmology()
    test_todevice_shape()
    print("All legacy tests passed!")
