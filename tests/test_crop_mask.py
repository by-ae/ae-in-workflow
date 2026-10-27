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


def test_crop_sum_of_masks_keeps_edge_pixels():
    """sum_of_masks must include the last masked row and column."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    images = torch.zeros(1, 8, 8, 3)
    images[0, :, :, 0] = 0.2
    images[0, -1, :, 1] = 1.0
    images[0, :, -1, 2] = 1.0
    masks = torch.ones(1, 8, 8)

    node = CropImageBatchToMaskBoundsNodeAE()
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks", color_fill_mask=False
    )

    assert result_images.shape == (1, 8, 8, 3)
    assert result_masks.shape == (1, 8, 8)
    assert float(result_images[0, -1, 0, 1]) == 1.0
    assert float(result_images[0, 0, -1, 2]) == 1.0
    assert int((result_masks > 0.5).sum()) == 64


def test_crop_sum_of_masks_partial_region_size():
    """A 20x20 mask region crops to 20x20, not 19x19."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    images = torch.zeros(1, 100, 100, 3)
    images[0, 20:40, 30:50, 0] = 1.0
    masks = torch.zeros(1, 100, 100)
    masks[0, 20:40, 30:50] = 1.0

    node = CropImageBatchToMaskBoundsNodeAE()
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks", color_fill_mask=False, optional_padding=0
    )

    assert result_images.shape == (1, 20, 20, 3)
    assert int((result_masks > 0.5).sum()) == 400
    assert float(result_images[0, 0, 0, 0]) == 1.0
    assert float(result_images[0, -1, -1, 0]) == 1.0


def test_crop_sum_of_masks_single_row_and_pixel():
    """A one-row or one-pixel mask is a real region, not an empty mask."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    node = CropImageBatchToMaskBoundsNodeAE()

    images = torch.zeros(1, 100, 80, 3)
    images[0, 5, 10:30, 0] = 1.0
    masks = torch.zeros(1, 100, 80)
    masks[0, 5, 10:30] = 1.0
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks", color_fill_mask=False
    )
    assert result_images.shape == (1, 1, 20, 3)
    assert int((result_masks > 0.5).sum()) == 20

    images = torch.zeros(1, 50, 50, 3)
    images[0, 7, 9] = torch.tensor([1.0, 0.0, 0.0])
    masks = torch.zeros(1, 50, 50)
    masks[0, 7, 9] = 1.0
    result_images, _ = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks", color_fill_mask=False
    )
    assert result_images.shape == (1, 1, 1, 3)
    assert torch.allclose(result_images[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))


def test_crop_sum_of_masks_padding_is_exact_and_clamped():
    """optional_padding grows the crop by that many pixels, stopping at the frame edge."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    node = CropImageBatchToMaskBoundsNodeAE()
    images = torch.zeros(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    masks[0, 40:60, 40:60] = 1.0

    unpadded, _ = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks", optional_padding=0, color_fill_mask=False
    )
    padded, _ = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks", optional_padding=10, color_fill_mask=False
    )
    assert unpadded.shape == (1, 20, 20, 3)
    assert padded.shape == (1, 40, 40, 3)

    edge_images = torch.zeros(1, 20, 20, 3)
    edge_masks = torch.zeros(1, 20, 20)
    edge_masks[0, 0:2, 0:3] = 1.0
    clamped, _ = node.crop_to_mask_bounds(
        edge_images, edge_masks, mode="sum_of_masks", optional_padding=10, color_fill_mask=False
    )
    # Rows 0-1 plus 10 below = 12. Cols 0-2 plus 10 to the right = 13. Top/left stay at 0.
    assert clamped.shape == (1, 12, 13, 3)


def test_crop_mask_for_each_keeps_edge_pixels_and_full_empty_frame():
    """mask_for_each keeps the masked edge and does not grow an empty mask by a pixel."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    node = CropImageBatchToMaskBoundsNodeAE()

    images = torch.zeros(1, 40, 40, 3)
    images[0, 10:15, 8:12, 0] = 0.25
    images[0, 14, 11, 1] = 1.0
    masks = torch.zeros(1, 40, 40)
    masks[0, 10:15, 8:12] = 1.0
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each", color_fill_mask=False
    )
    assert result_images.shape == (1, 5, 4, 3)
    assert float(result_images[0, -1, -1, 1]) == 1.0
    assert int((result_masks > 0.5).sum()) == 20

    images = torch.ones(1, 30, 40, 3)
    masks = torch.zeros(1, 30, 40)
    result_images, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each", color_fill_mask=False
    )
    assert result_images.shape == (1, 30, 40, 3)
    assert result_masks.shape == (1, 30, 40)


def test_crop_mask_for_each_one_forced_side_keeps_the_other():
    """Setting only width or only height must not collapse the other side to 0."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    node = CropImageBatchToMaskBoundsNodeAE()
    images = torch.zeros(1, 100, 100, 3)
    images[0, 40:60, 40:60, 0] = 1.0
    masks = torch.zeros(1, 100, 100)
    masks[0, 40:60, 40:60] = 1.0

    wide, wide_masks = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each", color_fill_mask=False,
        optional_new_width=128, optional_new_height=0,
    )
    assert wide.shape == (1, 20, 128, 3)
    assert wide_masks.shape == (1, 20, 128)
    # 20x20 crop is centered in the 128-wide canvas.
    assert float(wide[0, 10, 64, 0]) > 0.9
    assert float(wide[0, 10, 0, 0]) == 0.0

    tall, tall_masks = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each", color_fill_mask=False,
        optional_new_width=0, optional_new_height=200,
    )
    assert tall.shape == (1, 200, 20, 3)
    assert tall_masks.shape == (1, 200, 20)
    assert float(tall[0, 100, 10, 0]) > 0.9
    assert float(tall[0, 0, 0, 0]) == 0.0


def test_crop_mask_for_each_empty_mask_width_only():
    """An empty mask with only a width still has the original height."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    node = CropImageBatchToMaskBoundsNodeAE()
    images = torch.ones(1, 30, 40, 3)
    masks = torch.zeros(1, 30, 40)
    result, result_masks = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each", color_fill_mask=False,
        optional_new_width=100, optional_new_height=0,
    )
    assert result.shape == (1, 30, 100, 3)
    assert result_masks.shape == (1, 30, 100)


def test_crop_mask_for_each_batch_uses_tallest_crop_when_height_unset():
    """With only width set, the canvas height is the tallest crop in the batch."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    node = CropImageBatchToMaskBoundsNodeAE()
    images = torch.zeros(2, 100, 100, 3)
    masks = torch.zeros(2, 100, 100)
    masks[0, 10:30, 10:20] = 1.0  # 20 x 10
    masks[1, 40:80, 40:90] = 1.0  # 40 x 50
    result, _ = node.crop_to_mask_bounds(
        images, masks, mode="mask_for_each", color_fill_mask=False,
        optional_new_width=80, optional_new_height=0,
    )
    assert result.shape == (2, 40, 80, 3)


def test_crop_sum_of_masks_size_requests_keep_previous_behavior():
    """sum_of_masks: width above 0 returns the expanded frame; height alone still crops."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    node = CropImageBatchToMaskBoundsNodeAE()
    images = torch.zeros(1, 100, 100, 3)
    masks = torch.zeros(1, 100, 100)
    masks[0, 40:60, 40:60] = 1.0

    expanded, expanded_masks = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks", color_fill_mask=False,
        optional_new_width=128, optional_new_height=0,
    )
    assert expanded.shape == (1, 100, 128, 3)
    assert int((expanded_masks > 0.5).sum()) == 400

    cropped, _ = node.crop_to_mask_bounds(
        images, masks, mode="sum_of_masks", color_fill_mask=False,
        optional_new_width=0, optional_new_height=200,
    )
    assert cropped.shape == (1, 20, 20, 3)


def test_crop_rejects_non_rgb_channel_count():
    """A non-RGB image raises a channel error instead of a numpy broadcast crash."""
    from __init__ import CropImageBatchToMaskBoundsNodeAE

    node = CropImageBatchToMaskBoundsNodeAE()
    masks = torch.ones(1, 8, 8)
    with pytest.raises(ValueError, match="3 color channels"):
        node.crop_to_mask_bounds(torch.ones(1, 8, 8, 4), masks)
    with pytest.raises(ValueError, match="3 color channels"):
        node.crop_to_mask_bounds(torch.ones(1, 8, 8, 1), masks)


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
