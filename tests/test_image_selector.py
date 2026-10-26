"""
Tests for image_selector module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from PIL import Image


def test_module_imports():
    """Test that image_selector can be imported"""
    import image_selector
    assert hasattr(image_selector, 'hex_to_rgb')


def test_hex_to_rgb():
    """Test hex color conversion"""
    from image_selector import hex_to_rgb
    
    # 6-character format
    assert hex_to_rgb("#000000") == (0, 0, 0)
    assert hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert hex_to_rgb("#ff0000") == (255, 0, 0)
    assert hex_to_rgb("#00FF00") == (0, 255, 0)
    assert hex_to_rgb("#0000Ff") == (0, 0, 255)
    
    # 3-character format (should expand: #RGB -> #RRGGBB)
    assert hex_to_rgb("#000") == (0, 0, 0)
    assert hex_to_rgb("#FFF") == (255, 255, 255)
    assert hex_to_rgb("#F00") == (255, 0, 0)
    assert hex_to_rgb("#0F0") == (0, 255, 0)
    assert hex_to_rgb("#00F") == (0, 0, 255)
    assert hex_to_rgb("#abc") == (170, 187, 204)  # #aabbcc


def test_hex_to_rgb_invalid():
    """Test hex_to_rgb with invalid inputs"""
    from image_selector import hex_to_rgb
    
    with pytest.raises(ValueError):
        hex_to_rgb("invalid")
    
    with pytest.raises(ValueError):
        hex_to_rgb("#FFFF")  # 4 chars (invalid)
    
    with pytest.raises(ValueError):
        hex_to_rgb("FF0000")  # Missing #
    
    with pytest.raises(ValueError):
        hex_to_rgb("#FF")  # Too short (2 chars)


def test_create_thumbnail():
    """Test thumbnail creation with various aspect ratios"""
    from image_selector import create_thumbnail
    
    # Normal case - square image
    img = Image.new('RGB', (100, 100), color='red')
    result = create_thumbnail(img, (50, 50))
    assert result.size == (50, 50)
    
    # Wide image
    img = Image.new('RGB', (200, 100), color='blue')
    result = create_thumbnail(img, (50, 50))
    assert result.size[0] == 50
    assert result.size[1] <= 50
    
    # Tall image
    img = Image.new('RGB', (100, 200), color='green')
    result = create_thumbnail(img, (50, 50))
    assert result.size[0] <= 50
    assert result.size[1] == 50


def test_create_thumbnail_extreme_aspect_ratios():
    """Test thumbnail creation with extreme aspect ratios that could produce 0-pixel dimensions"""
    from image_selector import create_thumbnail
    
    # Very wide image (1000x1)
    img = Image.new('RGB', (1000, 1), color='blue')
    result = create_thumbnail(img, (150, 150))
    assert result.size[0] == 150
    assert result.size[1] >= 1  # Should be at least 1 pixel
    
    # Very tall image (1x1000)
    img = Image.new('RGB', (1, 1000), color='green')
    result = create_thumbnail(img, (150, 150))
    assert result.size[0] >= 1  # Should be at least 1 pixel
    assert result.size[1] == 150
    
    # 1x1 pixel image
    img = Image.new('RGB', (1, 1), color='red')
    result = create_thumbnail(img, (150, 150))
    assert result.size[0] >= 1
    assert result.size[1] >= 1


def test_pil_to_tensor():
    """Test PIL to tensor conversion"""
    from image_selector import pil_to_tensor
    
    # RGB image
    img = Image.new('RGB', (10, 10), color='red')
    tensor, mask = pil_to_tensor(img)
    assert tensor.shape == (1, 10, 10, 3)
    assert mask.shape == (1, 10, 10)
    assert torch.all(mask == 1.0)  # All-white mask for RGB
    
    # RGBA image with transparency
    img = Image.new('RGBA', (10, 10), color=(255, 0, 0, 128))
    tensor, mask = pil_to_tensor(img)
    assert tensor.shape == (1, 10, 10, 3)
    assert mask.shape == (1, 10, 10)
    assert torch.all(mask <= 1.0)
    assert torch.all(mask >= 0.0)


def test_pil_to_tensor_invalid():
    """Test pil_to_tensor with invalid inputs"""
    from image_selector import pil_to_tensor
    
    # None input
    with pytest.raises(ValueError, match="Image cannot be None"):
        pil_to_tensor(None)
    
    # Non-PIL object
    with pytest.raises(ValueError, match="Expected PIL Image"):
        pil_to_tensor("not an image")
    
    with pytest.raises(ValueError, match="Expected PIL Image"):
        pil_to_tensor(123)


def test_load_images_from_folder():
    """Test loading images from folder"""
    from image_selector import load_images_from_folder
    import tempfile
    
    # Empty folder
    with tempfile.TemporaryDirectory() as tmpdir:
        images, paths = load_images_from_folder(tmpdir)
        assert len(images) == 0
        assert len(paths) == 0
    
    # Non-existent folder
    images, paths = load_images_from_folder('/nonexistent/path')
    assert len(images) == 0
    assert len(paths) == 0
    
    # Empty string
    images, paths = load_images_from_folder('')
    assert len(images) == 0
    assert len(paths) == 0


def test_sort_image_paths():
    """Test sorting image paths"""
    from image_selector import sort_image_paths
    
    # Empty list
    result = sort_image_paths([], 'name', True)
    assert result == []
    
    # Single item
    result = sort_image_paths(['/path/to/image.jpg'], 'name', True)
    assert result == ['/path/to/image.jpg']
    
    # Name sorting
    paths = ['/path/c.jpg', '/path/a.jpg', '/path/b.jpg']
    result = sort_image_paths(paths, 'name', True)
    assert result == ['/path/a.jpg', '/path/b.jpg', '/path/c.jpg']
    
    # Descending
    result = sort_image_paths(paths, 'name', False)
    assert result == ['/path/c.jpg', '/path/b.jpg', '/path/a.jpg']


def test_build_folder_tree():
    """Test folder tree building"""
    from image_selector import build_folder_tree
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple folder structure
        os.makedirs(os.path.join(tmpdir, 'subfolder1'), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, 'subfolder2'), exist_ok=True)
        
        tree = build_folder_tree(tmpdir, tmpdir)
        
        # Should have at least the parent and current folder
        assert len(tree) >= 2
        
        # First item should be parent navigation
        assert tree[0][0].startswith('⬆️')
        
        # Second item should be current folder
        assert tree[1][0].startswith('📁')
    
    # Non-existent path
    tree = build_folder_tree('/nonexistent/path', '/nonexistent/path')
    assert len(tree) >= 2  # Should still have navigation items


def test_process_selected_images_empty():
    """Test process_selected_images with empty input"""
    from image_selector import process_selected_images
    
    images, masks, image_batch, mask_batch = process_selected_images([], [], None, None, '#000000')
    
    assert len(images) == 0
    assert len(masks) == 0
    assert image_batch.shape == (0, 0, 0, 3)
    assert mask_batch.shape == (0, 0, 0)


def test_lazy_image_loader():
    """Test LazyImageLoader class"""
    from image_selector import LazyImageLoader
    
    # Empty loader
    loader = LazyImageLoader([])
    assert loader.paths == []
    
    # Non-existent image
    loader = LazyImageLoader(['/nonexistent.jpg'])
    thumb = loader.get_thumbnail('/nonexistent.jpg')
    assert thumb is not None  # Should return error placeholder
    
    # Preload empty
    loader.preload_visible([])  # Should not crash


def test_pil_to_tensor_non_rgb_modes():
    """Palette, LA, 1-bit, and grayscale images become RGB tensors with a real mask."""
    from image_selector import pil_to_tensor

    indexed = Image.new('P', (5, 5), color=1)
    indexed.putpalette([0, 0, 0, 10, 20, 30] + [0] * (768 - 6))
    tensor, mask = pil_to_tensor(indexed)
    assert tensor.shape == (1, 5, 5, 3)
    assert torch.allclose(tensor[0, 0, 0], torch.tensor([10 / 255, 20 / 255, 30 / 255]), atol=1e-5)
    assert torch.all(mask == 1.0)

    la = Image.new('LA', (4, 4), color=(255, 64))
    tensor, mask = pil_to_tensor(la)
    assert tensor.shape == (1, 4, 4, 3)
    assert torch.allclose(tensor[0, 0, 0], torch.tensor([1.0, 1.0, 1.0]), atol=1e-5)
    assert torch.allclose(mask, torch.full((1, 4, 4), 64 / 255), atol=1e-4)

    bitmap = Image.new('1', (4, 4), color=1)
    tensor, mask = pil_to_tensor(bitmap)
    assert tensor.shape == (1, 4, 4, 3)
    assert torch.all(tensor == 1.0)
    assert torch.all(mask == 1.0)

    gray = Image.new('L', (8, 6), color=200)
    tensor, mask = pil_to_tensor(gray)
    assert tensor.shape == (1, 6, 8, 3)
    assert torch.allclose(tensor[0, 0, 0], torch.tensor([200 / 255, 200 / 255, 200 / 255]), atol=1e-5)


def test_process_selected_images_extreme_aspect_and_target_sides():
    """Very wide images resize, and a single target side is honored."""
    from image_selector import process_selected_images

    wide = Image.new('RGB', (10000, 3), (0, 0, 255))
    _, _, batch, _ = process_selected_images([wide], [0], None, None, '#000000')
    assert batch.shape[0] == 1
    assert batch.shape[1] >= 1
    assert batch.shape[2] == 2048
    assert batch.shape[3] == 3

    photo = Image.new('RGB', (100, 50), 'red')
    _, _, batch, _ = process_selected_images([photo], [0], 80, None, '#000000')
    assert batch.shape == (1, 40, 80, 3)
    _, _, batch, _ = process_selected_images([photo], [0], None, 40, '#000000')
    assert batch.shape == (1, 40, 80, 3)
    _, _, batch, _ = process_selected_images([photo], [0], 0, 0, '#000000')
    assert batch.shape == (1, 50, 100, 3)
    _, _, batch, _ = process_selected_images([photo], [0], 40, 20, '#000000')
    assert batch.shape == (1, 20, 40, 3)

    small = Image.new('RGB', (10, 10), 'red')
    large = Image.new('RGB', (100, 50), 'blue')
    _, _, batch, _ = process_selected_images([small, large], [0, 1], 80, None, '#000000')
    assert batch.shape == (2, 40, 80, 3)
