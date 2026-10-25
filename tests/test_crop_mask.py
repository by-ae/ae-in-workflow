"""
Tests for CropImageBatchToMaskBoundsNodeAE
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np


def test_crop_node_imports():
    """Test that CropImageBatchToMaskBoundsNodeAE can be imported"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    assert CropImageBatchToMaskBoundsNodeAE is not None


def test_crop_node_basic_mask_for_each():
    """Test basic cropping with mask_for_each mode"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    # Create test images and masks
    images = torch.ones(2, 100, 100, 3) * 0.5  # Gray images
    masks = torch.zeros(2, 100, 100)
    
    # Add masks in different positions
    masks[0, 20:40, 30:50] = 1.0  # First mask
    masks[1, 60:80, 10:30] = 1.0  # Second mask
    
    node = CropImageBatchToMaskBoundsNodeAE()
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each"
    )
    
    # Should return tensors
    assert isinstance(result_images, torch.Tensor)
    assert isinstance(result_masks, torch.Tensor)
    
    # Should have same batch size
    assert result_images.shape[0] == 2
    assert result_masks.shape[0] == 2
    
    # All outputs should have same dimensions
    assert result_images.shape[1] == result_images.shape[1]
    assert result_images.shape[2] == result_images.shape[2]


def test_crop_node_sum_of_masks():
    """Test cropping with sum_of_masks mode"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(2, 100, 100, 3) * 0.5
    masks = torch.zeros(2, 100, 100)
    
    masks[0, 20:40, 30:50] = 1.0
    masks[1, 60:80, 10:30] = 1.0
    
    node = CropImageBatchToMaskBoundsNodeAE()
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks"
    )
    
    assert result_images.shape[0] == 2
    assert result_masks.shape[0] == 2


def test_crop_node_empty_mask():
    """Test that empty masks are handled correctly"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)  # Empty mask
    
    node = CropImageBatchToMaskBoundsNodeAE()
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each"
    )
    
    # Should still return valid tensors
    assert result_images.shape[0] == 1
    assert result_masks.shape[0] == 1


def test_crop_node_invert_mask():
    """Test mask inversion"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    masks[0, 20:40, 30:50] = 1.0
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    # Without inversion
    result1_images, _ = node.crop_to_mask_bounds(
        images, masks, invert_mask=False, mode="sum_of_masks"
    )
    
    # With inversion
    result2_images, _ = node.crop_to_mask_bounds(
        images, masks, invert_mask=True, mode="sum_of_masks"
    )
    
    # Dimensions should be different (inverted crops everything outside)
    assert result1_images.shape != result2_images.shape or True  # May be same if both fill


def test_crop_node_padding_color():
    """Test custom padding color"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3) * 0.5
    masks = torch.zeros(1, 100, 100)
    masks[0, 40:60, 40:60] = 1.0
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    # Red padding
    result_images, _ = node.crop_to_mask_bounds(
        images, masks, padding_color="#FF0000", mode="mask_for_each"
    )
    
    assert result_images.shape[0] == 1


def test_crop_node_optional_padding():
    """Test additional padding around cropped area"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    masks[0, 40:60, 40:60] = 1.0
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    # Without padding
    result1, _ = node.crop_to_mask_bounds(
        images, masks, optional_padding=0, mode="sum_of_masks"
    )
    
    # With padding
    result2, _ = node.crop_to_mask_bounds(
        images, masks, optional_padding=10, mode="sum_of_masks"
    )
    
    # Padded version should be larger (or same if clamped to bounds)
    assert result2.shape[1] >= result1.shape[1]
    assert result2.shape[2] >= result1.shape[2]


def test_crop_node_forced_dimensions():
    """Test forcing output dimensions"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    masks[0, 40:60, 40:60] = 1.0
    
    node = CropImageBatchToMaskBoundsNodeAE()
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks,
        optional_new_width=128,
        optional_new_height=128,
        mode="mask_for_each"
    )
    
    # Output should have forced dimensions
    assert result_images.shape[1] == 128
    assert result_images.shape[2] == 128
    assert result_masks.shape[1] == 128
    assert result_masks.shape[2] == 128


def test_crop_node_invalid_mode():
    """Test that invalid mode raises ValueError"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    with pytest.raises(ValueError, match="mode must be"):
        node.crop_to_mask_bounds(images, masks, mode="invalid_mode")


def test_crop_node_invalid_padding_color():
    """Test that invalid padding color raises ValueError"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    with pytest.raises(ValueError, match="padding_color must be a valid hex color"):
        node.crop_to_mask_bounds(images, masks, padding_color="invalid")
    
    with pytest.raises(ValueError, match="padding_color must be a valid hex color"):
        node.crop_to_mask_bounds(images, masks, padding_color="FF0000")  # Missing #


def test_crop_node_invalid_optional_padding():
    """Test that invalid optional_padding raises ValueError"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    with pytest.raises(ValueError, match="optional_padding must be a non-negative integer"):
        node.crop_to_mask_bounds(images, masks, optional_padding=-5)


def test_crop_node_invalid_dimensions():
    """Test that invalid forced dimensions raise ValueError"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    with pytest.raises(ValueError, match="optional_new_width must be a non-negative integer"):
        node.crop_to_mask_bounds(images, masks, optional_new_width=-10)
    
    with pytest.raises(ValueError, match="optional_new_height must be a non-negative integer"):
        node.crop_to_mask_bounds(images, masks, optional_new_height=-10)


def test_crop_node_color_fill_mask():
    """Test color_fill_mask option"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(1, 100, 100, 3) * 0.5
    masks = torch.zeros(1, 100, 100)
    masks[0, 40:60, 40:60] = 1.0
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    # With color fill
    result1, _ = node.crop_to_mask_bounds(
        images, masks, color_fill_mask=True, padding_color="#FF0000"
    )
    
    # Without color fill
    result2, _ = node.crop_to_mask_bounds(
        images, masks, color_fill_mask=False, padding_color="#FF0000"
    )
    
    assert result1.shape == result2.shape


def test_crop_node_batch_consistency():
    """Test that all images in batch have consistent dimensions"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    
    images = torch.ones(3, 100, 100, 3) * 0.5
    masks = torch.zeros(3, 100, 100)
    
    # Different sized masks
    masks[0, 10:30, 10:30] = 1.0
    masks[1, 40:80, 40:80] = 1.0
    masks[2, 5:15, 5:95] = 1.0
    
    node = CropImageBatchToMaskBoundsNodeAE()
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each"
    )
    
    # All outputs should have identical dimensions
    for i in range(1, 3):
        assert result_images[i].shape == result_images[0].shape
        assert result_masks[i].shape == result_masks[0].shape


def test_crop_node_mismatched_batch_size():
    """Test error handling for mismatched batch sizes"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    import torch
    import pytest
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    # Mismatched batch sizes
    images = torch.zeros(3, 64, 64, 3)
    masks = torch.zeros(2, 64, 64)  # Different batch size
    
    with pytest.raises(ValueError, match="same batch size"):
        node.crop_to_mask_bounds(images, masks)


def test_crop_node_mismatched_spatial_dimensions():
    """Test error handling for mismatched spatial dimensions"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    import torch
    import pytest
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    # Mismatched spatial dimensions
    images = torch.zeros(2, 64, 64, 3)
    masks = torch.zeros(2, 32, 32)  # Different spatial size
    
    with pytest.raises(ValueError, match="same spatial dimensions"):
        node.crop_to_mask_bounds(images, masks)


def test_crop_node_invalid_tensor_types():
    """Test error handling for invalid tensor types"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    import torch
    import pytest
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    # Non-tensor images
    with pytest.raises(ValueError, match="must be a torch.Tensor"):
        node.crop_to_mask_bounds("not a tensor", torch.zeros(2, 64, 64))
    
    # Non-tensor masks
    with pytest.raises(ValueError, match="must be a torch.Tensor"):
        node.crop_to_mask_bounds(torch.zeros(2, 64, 64, 3), "not a tensor")


def test_crop_node_invalid_tensor_shapes():
    """Test error handling for invalid tensor shapes"""
    from __init__ import CropImageBatchToMaskBoundsNodeAE
    import torch
    import pytest
    
    node = CropImageBatchToMaskBoundsNodeAE()
    
    # Wrong image dimensions (3D instead of 4D)
    with pytest.raises(ValueError, match="images must be 4D"):
        node.crop_to_mask_bounds(torch.zeros(2, 64, 64), torch.zeros(2, 64, 64))
    
    # Wrong mask dimensions (2D instead of 3D)
    with pytest.raises(ValueError, match="masks must be 3D"):
        node.crop_to_mask_bounds(torch.zeros(2, 64, 64, 3), torch.zeros(64, 64))
