"""
Tests for zimage_i2l module
"""
import sys
import os

# Add parent directory to path for direct module import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pytest


def test_module_imports():
    """Test that zimage_i2l can be imported"""
    import zimage_i2l
    assert hasattr(zimage_i2l, 'DIFF_SYNTH_AVAILABLE')
    assert hasattr(zimage_i2l, 'DEPENDENCIES_AVAILABLE')


def test_sanitize_lora_name():
    """Test lora name sanitization"""
    from zimage_i2l import sanitize_lora_name
    
    assert sanitize_lora_name("mylora") == "mylora"
    assert sanitize_lora_name("MyLoRA") == "mylora"
    assert sanitize_lora_name("my lora name") == "my_lora_name"
    assert sanitize_lora_name("my-lora!@#") == "mylora"
    assert sanitize_lora_name("My LoRA 2024!") == "my_lora_2024"
    assert sanitize_lora_name("") == ""


def test_find_next_version(tmp_path):
    """Test version number finding"""
    from zimage_i2l import find_next_version
    
    base_path = tmp_path / "test_lora"
    version = find_next_version(str(base_path))
    assert version == "000"
    
    # Create v000
    (tmp_path / "test_lora_v000.safetensors").touch()
    version = find_next_version(str(base_path))
    assert version == "001"
    
    # Create v001, v002
    for i in range(1, 3):
        (tmp_path / f"test_lora_v{i:03d}.safetensors").touch()
    version = find_next_version(str(base_path))
    assert version == "003"


def test_process_images_to_lora_without_deps():
    """Test that process_images_to_lora raises informative error without deps"""
    from zimage_i2l import DIFF_SYNTH_AVAILABLE, process_images_to_lora
    
    if not DIFF_SYNTH_AVAILABLE:
        import torch
        dummy_tensor = torch.zeros(1, 64, 64, 3)
        
        with pytest.raises(ImportError) as exc_info:
            process_images_to_lora(dummy_tensor, "test", 1)
        
        assert "DiffSynth-Studio" in str(exc_info.value)
