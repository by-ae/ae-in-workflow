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


def test_reduce_loras_by_batching_empty():
    """Test reduce_loras_by_batching with empty input"""
    from zimage_i2l import reduce_loras_by_batching
    
    def dummy_merge(loras, weights):
        return "merged"
    
    result_loras, result_weights = reduce_loras_by_batching([], [], 2, dummy_merge)
    assert result_loras == []
    assert result_weights == []


def test_reduce_loras_by_batching_factor_one():
    """Test reduce_loras_by_batching with factor=1 (no-op)"""
    from zimage_i2l import reduce_loras_by_batching
    
    loras = ["lora1", "lora2", "lora3"]
    weights = [0.5, 1.0, 0.3]
    
    def dummy_merge(loras, weights):
        return "merged"
    
    result_loras, result_weights = reduce_loras_by_batching(loras, weights, 1, dummy_merge)
    assert result_loras == loras
    assert result_weights == weights


def test_reduce_loras_by_batching_factor_two():
    """Test reduce_loras_by_batching with factor=2"""
    from zimage_i2l import reduce_loras_by_batching
    
    loras = ["lora1", "lora2", "lora3"]
    weights = [0.5, 1.0, 0.3]
    
    merge_calls = []
    def track_merge(loras, weights):
        merge_calls.append((loras, weights))
        return f"merged_{len(merge_calls)}"
    
    result_loras, result_weights = reduce_loras_by_batching(loras, weights, 2, track_merge)
    
    # Should batch into 2 groups: [1.0, 0.5] and [0.3]
    assert len(result_loras) == 2
    assert len(result_weights) == 2
    
    # First batch should be merged
    assert result_loras[0] == "merged_1"
    assert result_weights[0] == 1.5  # 1.0 + 0.5
    
    # Second batch has only one element, so passes through
    assert result_loras[1] == "lora3"
    assert result_weights[1] == 0.3


def test_reduce_loras_by_batching_factor_three():
    """Test reduce_loras_by_batching with factor=3 and uneven count"""
    from zimage_i2l import reduce_loras_by_batching
    
    loras = ["lora1", "lora2", "lora3", "lora4", "lora5"]
    weights = [0.8, 0.5, 1.0, 0.2, 0.6]
    
    merge_calls = []
    def track_merge(loras, weights):
        merge_calls.append((loras, weights))
        return f"merged_{len(merge_calls)}"
    
    result_loras, result_weights = reduce_loras_by_batching(loras, weights, 3, track_merge)
    
    # Should batch into 2 groups: [1.0, 0.8, 0.6] and [0.5, 0.2]
    assert len(result_loras) == 2
    assert len(result_weights) == 2
    
    # First batch: top 3 weights merged
    assert result_loras[0] == "merged_1"
    assert result_weights[0] == 2.4  # 1.0 + 0.8 + 0.6
    
    # Second batch: remaining 2 merged
    assert result_loras[1] == "merged_2"
    assert result_weights[1] == 0.7  # 0.5 + 0.2


def test_reduce_loras_by_batching_ordering():
    """Test that reduce_loras_by_batching sorts by weight descending"""
    from zimage_i2l import reduce_loras_by_batching
    
    loras = ["low", "high", "medium"]
    weights = [0.1, 1.0, 0.5]
    
    captured_order = []
    def track_merge(loras, weights):
        captured_order.extend(loras)
        return "merged"
    
    reduce_loras_by_batching(loras, weights, 10, track_merge)
    
    # Should process in descending weight order: high (1.0), medium (0.5), low (0.1)
    assert captured_order == ["high", "medium", "low"]


def test_reduce_loras_by_batching_weight_sums_preserved():
    """Test that total weight is preserved after batching"""
    from zimage_i2l import reduce_loras_by_batching
    
    loras = ["a", "b", "c", "d"]
    weights = [0.25, 0.5, 0.75, 1.0]
    
    def dummy_merge(loras, weights):
        return "merged"
    
    result_loras, result_weights = reduce_loras_by_batching(loras, weights, 2, dummy_merge)
    
    # Total weight should be preserved
    assert sum(weights) == sum(result_weights)


def test_merge_lora_weighted_average():
    """Test merge_lora_weighted_average with torch tensors"""
    torch = pytest.importorskip("torch")
    from zimage_i2l import merge_lora_weighted_average
    
    # Create dummy LoRA dicts with lora_A and lora_B tensors
    lora1 = {
        "layer1.lora_A.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer1.lora_B.weight": torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
    }
    lora2 = {
        "layer1.lora_A.weight": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        "layer1.lora_B.weight": torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
    }
    
    weights = [0.3, 0.7]
    result = merge_lora_weighted_average([lora1, lora2], weights)
    
    # Check weighted average: (1.0 * 0.3 + 2.0 * 0.7) / 1.0 = 1.7
    expected_A = torch.tensor([[1.7, 2.7], [3.7, 4.7]])
    expected_B = torch.tensor([[0.85, 1.35], [1.85, 2.35]])
    
    assert torch.allclose(result["layer1.lora_A.weight"], expected_A, atol=1e-6)
    assert torch.allclose(result["layer1.lora_B.weight"], expected_B, atol=1e-6)


def test_merge_lora_weighted_average_uniform_fallback():
    """Test merge_lora_weighted_average fallback when weights sum to 0"""
    torch = pytest.importorskip("torch")
    from zimage_i2l import merge_lora_weighted_average
    
    lora1 = {
        "layer1.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "layer1.lora_B.weight": torch.tensor([[0.5, 1.0]]),
    }
    lora2 = {
        "layer1.lora_A.weight": torch.tensor([[3.0, 4.0]]),
        "layer1.lora_B.weight": torch.tensor([[1.5, 2.0]]),
    }
    
    # All weights are 0
    weights = [0.0, 0.0]
    result = merge_lora_weighted_average([lora1, lora2], weights)
    
    # Should fall back to uniform: (1.0 + 3.0) / 2 = 2.0
    expected_A = torch.tensor([[2.0, 3.0]])
    expected_B = torch.tensor([[1.0, 1.5]])
    
    assert torch.allclose(result["layer1.lora_A.weight"], expected_A, atol=1e-6)
    assert torch.allclose(result["layer1.lora_B.weight"], expected_B, atol=1e-6)


def test_merge_lora_weighted_average_empty():
    """Test merge_lora_weighted_average with empty input"""
    from zimage_i2l import merge_lora_weighted_average
    
    result = merge_lora_weighted_average([], [])
    assert result == {}


def test_process_images_to_lora_without_deps():
    """Test that process_images_to_lora raises informative error without deps"""
    from zimage_i2l import DIFF_SYNTH_AVAILABLE, process_images_to_lora
    
    if not DIFF_SYNTH_AVAILABLE:
        torch = pytest.importorskip("torch")
        dummy_tensor = torch.zeros(1, 64, 64, 3)
        
        with pytest.raises(ImportError) as exc_info:
            process_images_to_lora(dummy_tensor, "test", 1)
        
        assert "DiffSynth-Studio" in str(exc_info.value)


def test_unload_models_before_running_is_honored(monkeypatch):
    """The unload switch must be able to leave already loaded models in place."""
    import __init__ as nodes

    torch = pytest.importorskip("torch")

    class FakeModelManagement:
        def __init__(self):
            self.unloaded = 0

        def unload_all_models(self):
            self.unloaded += 1

        def soft_empty_cache(self, _force):
            pass

    fake = FakeModelManagement()
    monkeypatch.setattr(nodes, "model_management", fake)
    monkeypatch.setattr(nodes, "DIFF_SYNTH_AVAILABLE", True)
    monkeypatch.setattr(nodes, "DEPENDENCIES_AVAILABLE", True)
    monkeypatch.setattr(
        nodes,
        "process_images_to_lora",
        lambda *args, **kwargs: "Z-Image/ae/z-image_test_v000.safetensors",
    )

    node = nodes.ZImageImagesToLoRANodeAE()
    images = torch.zeros(1, 4, 4, 3)

    kept = node.images_to_lora(images, "test", unload_models_before_running=False)
    assert fake.unloaded == 0
    assert kept == ("Z-Image/ae/z-image_test_v000.safetensors",)

    node.images_to_lora(images, "test", unload_models_before_running=True)
    assert fake.unloaded == 1

