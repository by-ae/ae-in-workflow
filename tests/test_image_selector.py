"""
Tests for image_selector module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_module_imports():
    """Test that image_selector can be imported"""
    import image_selector
    assert hasattr(image_selector, 'hex_to_rgb')


def test_hex_to_rgb():
    """Test hex color conversion"""
    from image_selector import hex_to_rgb
    
    assert hex_to_rgb("#000000") == (0, 0, 0)
    assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert hex_to_rgb("#ff0000") == (255, 0, 0)
    assert hex_to_rgb("#00FF00") == (0, 255, 0)
    assert hex_to_rgb("#0000Ff") == (0, 0, 255)


def test_hex_to_rgb_invalid():
    """Test hex_to_rgb with invalid inputs"""
    from image_selector import hex_to_rgb
    
    with pytest.raises(ValueError):
        hex_to_rgb("invalid")
    
    with pytest.raises(ValueError):
        hex_to_rgb("#FFFF")
    
    with pytest.raises(ValueError):
        hex_to_rgb("FF0000")  # Missing #
    
    with pytest.raises(ValueError):
        hex_to_rgb("#F0F")  # Too short
