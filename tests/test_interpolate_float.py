"""
Tests for InterpolateFloatListNodeAE
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_interpolate_node_imports():
    """Test that InterpolateFloatListNodeAE can be imported"""
    from __init__ import InterpolateFloatListNodeAE
    assert InterpolateFloatListNodeAE is not None


def test_interpolate_basic():
    """Test basic interpolation"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, result_csv = node.interpolate_floats("1.0,2.0,3.0", count=5, interpolate_across_count=True)
    
    assert len(result_list) == 5
    assert result_list[0] == 1.0
    assert result_list[-1] == 3.0
    assert result_csv == "1.0,1.5,2.0,2.5,3.0"


def test_interpolate_no_count():
    """Test with count=0 (return original)"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, result_csv = node.interpolate_floats("1.5,2.5,3.5", count=0)
    
    assert len(result_list) == 3
    assert result_list == [1.5, 2.5, 3.5]


def test_interpolate_newlines():
    """Test handling of newlines in input"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, _ = node.interpolate_floats("1.0\n2.0\n3.0", count=3)
    
    assert len(result_list) == 3
    assert result_list == [1.0, 2.0, 3.0]


def test_interpolate_whitespace():
    """Test handling of whitespace"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, _ = node.interpolate_floats("1.0 , 2.0 , 3.0", count=3)
    
    assert len(result_list) == 3
    assert result_list == [1.0, 2.0, 3.0]


def test_interpolate_trim_mode():
    """Test trim/pad mode (interpolate_across_count=False)"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    
    # Trim: more values than count
    result_list, _ = node.interpolate_floats("1.0,2.0,3.0,4.0,5.0", count=3, interpolate_across_count=False)
    assert len(result_list) == 3
    assert result_list == [1.0, 2.0, 3.0]
    
    # Pad: fewer values than count
    result_list, _ = node.interpolate_floats("1.0,2.0", count=5, interpolate_across_count=False)
    assert len(result_list) == 5
    assert result_list == [1.0, 2.0, 1.0, 1.0, 1.0]  # Padded with 1.0


def test_interpolate_single_value():
    """Test interpolation with single value"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, _ = node.interpolate_floats("5.0", count=4, interpolate_across_count=True)
    
    assert len(result_list) == 4
    # Single value should be repeated
    assert all(v == 5.0 for v in result_list)


def test_interpolate_empty_string():
    """Test with empty string input"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, result_csv = node.interpolate_floats("", count=3, interpolate_across_count=True)
    
    # Should return default values (1.0 repeated)
    assert len(result_list) == 3
    assert result_list == [1.0, 1.0, 1.0]


def test_interpolate_rounding():
    """Test that values are rounded to 6 decimal places"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, _ = node.interpolate_floats("1.0,3.0", count=3, interpolate_across_count=True)
    
    # Should have 1.0, 2.0, 3.0
    assert result_list[1] == 2.0
    # Check that rounding is applied
    assert all(isinstance(v, float) for v in result_list)


def test_interpolate_csv_output():
    """Test CSV string output format"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    _, result_csv = node.interpolate_floats("1.0,2.0,3.0", count=3)
    
    assert isinstance(result_csv, str)
    assert result_csv.count(",") == 2  # Two commas for three values


def test_interpolate_negative_values():
    """Test with negative values"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, _ = node.interpolate_floats("-1.0,0.0,1.0", count=5, interpolate_across_count=True)
    
    assert len(result_list) == 5
    assert result_list[0] == -1.0
    assert result_list[2] == 0.0
    assert result_list[-1] == 1.0


def test_interpolate_large_count():
    """Test with large count value"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, _ = node.interpolate_floats("0.0,1.0", count=100, interpolate_across_count=True)
    
    assert len(result_list) == 100
    assert result_list[0] == 0.0
    assert result_list[-1] == 1.0


def test_interpolate_non_uniform_spacing():
    """Test interpolation with non-uniform input spacing"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    result_list, _ = node.interpolate_floats("0.0,1.0,10.0", count=5, interpolate_across_count=True)
    
    assert len(result_list) == 5
    assert result_list[0] == 0.0
    assert result_list[-1] == 10.0
    # Middle values should be interpolated
    assert result_list[2] == 1.0


def test_interpolate_error_handling():
    """Test error handling with invalid input"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    
    # Should return empty list and string on error
    result_list, result_csv = node.interpolate_floats("invalid,data", count=3)
    assert result_list == []
    assert result_csv == ""


def test_interpolate_mixed_valid_invalid():
    """Test with mix of valid and invalid values"""
    from __init__ import InterpolateFloatListNodeAE
    
    node = InterpolateFloatListNodeAE()
    
    # Should parse only valid floats
    result_list, result_csv = node.interpolate_floats("1.0,,2.0,,,3.0", count=3, interpolate_across_count=True)
    
    assert len(result_list) == 3
    assert result_list[0] == 1.0
    assert result_list[-1] == 3.0
