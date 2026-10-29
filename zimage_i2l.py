# Check for DiffSynth-Studio availability
from typing import Any, List, Tuple, Optional, Callable
import math
import re
import os


try:
    from diffsynth.pipelines.z_image import (
        ZImagePipeline, ModelConfig,
        ZImageUnit_Image2LoRAEncode, ZImageUnit_Image2LoRADecode
    )
    from diffsynth.utils.lora import merge_lora
    DIFF_SYNTH_AVAILABLE = True
except ImportError:
    DIFF_SYNTH_AVAILABLE = False

# Check for other required dependencies
try:
    from safetensors.torch import save_file
    import torch
    from PIL import Image
    import folder_paths
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# Overall availability
ZIMAGE_AVAILABLE = DIFF_SYNTH_AVAILABLE and DEPENDENCIES_AVAILABLE


def sanitize_lora_name(name: str) -> str:
    """Keep letters and digits, lowercase, and turn spaces into underscores.

    Letters from any language count. Other characters, including underscores
    and punctuation, are removed so the result can be used in a filename.
    Leading and trailing space is removed before spaces are collapsed. An
    empty string means the name had no letters or digits.
    """
    if not isinstance(name, str):
        raise ValueError(f"lora_name must be a string, got {type(name).__name__}")
    kept = "".join(ch for ch in name if ch.isalnum() or ch.isspace())
    kept = kept.strip().lower()
    return re.sub(r"\s+", "_", kept)


def require_lora_filename(name: str) -> str:
    """Return a non-empty filename stem, or raise ValueError."""
    sanitized = sanitize_lora_name(name)
    if not sanitized:
        raise ValueError(
            "lora_name needs at least one letter or number. "
            f"Got {name!r}, which is empty after removing characters that cannot go in a filename."
        )
    return sanitized


def parse_lora_weights(lora_weights: str) -> List[float]:
    """Parse comma-separated image weights. An empty string means equal weights.

    A token that is not a finite number raises ValueError and names that
    token. The node used to ignore the whole list and save equal weights.
    """
    if not isinstance(lora_weights, str):
        raise ValueError(
            "lora_weights must be a comma-separated string of numbers, "
            f"got {type(lora_weights).__name__}"
        )
    if not lora_weights.strip():
        return []
    weights = []
    invalid = []
    for token in lora_weights.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            value = float(token)
        except ValueError:
            invalid.append(token)
            continue
        if not math.isfinite(value):
            invalid.append(token)
            continue
        weights.append(value)
    if invalid:
        shown = ", ".join(invalid[:5])
        raise ValueError(
            f"lora_weights must be comma-separated numbers; could not parse: {shown}"
        )
    return weights

def find_next_version(base_path: str) -> str:
    """Find the next available version number (000, 001, 002, etc.)
    
    Args:
        base_path: Base path for the lora file (without version suffix)
        
    Returns:
        Three-digit version string (e.g. '000', '001')
    """
    version = 0
    while True:
        version_str = f"{version:03d}"
        test_path = f"{base_path}_v{version_str}.safetensors"
        if not os.path.exists(test_path):
            return version_str
        version += 1


def reduce_loras_by_batching(
    loras: List[Any],
    weights: List[float],
    factor: int,
    merge_fn: Callable[[List[Any], List[float]], Any]
) -> Tuple[List[Any], List[float]]:
    """Reduce size by averaging LoRAs in batches of similar weights.
    
    This is a pure function that does not depend on DiffSynth being installed.
    
    Args:
        loras: List of LoRA objects to batch
        weights: Corresponding weights for each LoRA
        factor: Batch size (number of LoRAs to merge together)
        merge_fn: Function to merge a batch of loras with weights
        
    Returns:
        Tuple of (batched_loras, batched_weights)
    """
    if not loras:
        return [], []
    
    if factor <= 1:
        return loras, weights
    
    # Sort by weight descending to group similar weights together
    sorted_pairs = sorted(zip(weights, loras), key=lambda x: x[0], reverse=True)
    sorted_weights, sorted_loras = zip(*sorted_pairs)
    
    batch_size = factor
    num_loras = len(sorted_loras)
    
    batched_loras = []
    batched_weights = []
    
    for i in range(0, num_loras, batch_size):
        batch_end = min(i + batch_size, num_loras)
        batch_loras = list(sorted_loras[i:batch_end])
        batch_weights = list(sorted_weights[i:batch_end])
        
        if len(batch_loras) == 1:
            # Single LoRA passes through unchanged
            batched_loras.append(batch_loras[0])
            batched_weights.append(batch_weights[0])
        else:
            # Merge batch using provided merge function
            averaged_lora = merge_fn(batch_loras, batch_weights)
            batched_loras.append(averaged_lora)
            # Preserve total batch strength (sum, not avg) for consistency
            batched_weights.append(sum(batch_weights))
    
    return batched_loras, batched_weights


def merge_lora_weighted_average(loras: List[dict], weights: List[float]) -> dict:
    """Average LoRAs with weights instead of concatenating - memory-efficient version"""
    if not loras:
        return {}
    
    merged = {}
    keys = [k for k in loras[0].keys() if ".lora_A." in k]
    
    # Precompute sum of weights once (scalar)
    sum_weights = sum(weights)
    use_uniform = sum_weights <= 0
    if use_uniform:
        sum_weights = len(weights)  # fallback to uniform
    
    for key in keys:
        # NOTE: Fixed - use torch.zeros_like (correct API)
        first_A = loras[0][key]
        merged[key] = torch.zeros_like(first_A)          # same device, dtype, shape
        lora_B_key = key.replace(".lora_A.", ".lora_B.")
        first_B = loras[0][lora_B_key]
        merged[lora_B_key] = torch.zeros_like(first_B)
        
        for i, lora in enumerate(loras):
            if use_uniform:
                # Uniform weight: 1.0 / len(weights)
                scale = 1.0 / sum_weights
            else:
                w = weights[i]
                if w == 0:
                    continue
                scale = w / sum_weights
            merged[key] += lora[key] * scale
            merged[lora_B_key] += lora[lora_B_key] * scale
    
    return merged


if DIFF_SYNTH_AVAILABLE:
    class CustomZImageUnit_Image2LoRADecode(ZImageUnit_Image2LoRADecode):
        def __init__(self, lora_weights=None, normalized_strength=1.0, reduce_size_factor=1):
            super().__init__()
            self.lora_weights = lora_weights or []
            self.normalized_strength = normalized_strength
            self.reduce_size_factor = reduce_size_factor  # Batch size for smart averaging
        
        def process(self, pipe, image2lora_x):
            if image2lora_x is None:
                return {}
            
            loras = []
            if pipe.image2lora_style is not None:
                pipe.load_models_to_device(["image2lora_style"])
                for x in image2lora_x:
                    loras.append(pipe.image2lora_style(x=x, residual=None))
            
            # Apply custom weighting with strength factor
            if self.lora_weights and loras:
                num_loras = len(loras)
                
                padded_weights = self.lora_weights[:num_loras] + [1.0] * (num_loras - len(self.lora_weights))
                
                # Sum to normalized_strength normalization
                total_weight = sum(padded_weights)
                if total_weight > 0:
                    final_weights = [w * self.normalized_strength / total_weight for w in padded_weights]
                else:
                    final_weights = [self.normalized_strength / num_loras] * num_loras
                
                # Apply size reduction if requested
                if self.reduce_size_factor > 1:
                    loras, final_weights = self._reduce_size_by_batching(loras, final_weights)
                
                
                lora = merge_lora_weighted_average(loras, final_weights)
            
            else:
                # No custom weights - handle size reduction if requested
                if self.reduce_size_factor > 1:
                    # Apply normalized_strength to equal weights
                    n = len(loras)
                    equal_weights = [self.normalized_strength / n] * n if n > 0 else []
                    loras, batched_weights = self._reduce_size_by_batching(loras, equal_weights)
                    lora = merge_lora_weighted_average(loras, batched_weights)
                else:
                    # Apply normalized_strength to normal weighting
                    n = len(loras)
                    alpha = self.normalized_strength / n if n > 0 else 1.0
                    lora = merge_lora(loras, alpha=alpha)
            
            return {"lora": lora}
        
        def _reduce_size_by_batching(self, loras, weights):
            """Reduce size by averaging LoRAs in batches of similar weights"""
            batched_loras, batched_weights = reduce_loras_by_batching(
                loras, weights, self.reduce_size_factor, merge_lora_weighted_average
            )
            if len(batched_loras) < len(loras):
                print(f"Smart size reduction: {len(loras)} → {len(batched_loras)} LoRAs (factor: {self.reduce_size_factor})")
            return batched_loras, batched_weights

    def process_images_to_lora(images_tensor, lora_name, batch_size, lora_weights=None, normalized_strength=1.0, reduce_size_factor=1):
        if lora_weights is None:
            lora_weights = []
        # Reject a name that would save as z-image__vNNN before any model work.
        sanitized_name = require_lora_filename(lora_name)
        # Convert tensor to PIL images (assuming shape N, H, W, 3)
        images = []
        for i in range(images_tensor.shape[0]):
            # Convert tensor to PIL Image
            img_tensor = images_tensor[i]  # Shape: (H, W, 3)
            img_array = (img_tensor * 255).clamp(0, 255).byte().cpu().numpy()
            img = Image.fromarray(img_array)
            images.append(img)

        # Setup folders and paths using ComfyUI's folder_paths system
        lora_folders = folder_paths.get_folder_paths("loras")
        lora_folder = lora_folders[0] if lora_folders else 'models/loras/'
        save_location_base = f'Z-Image/ae/z-image_{sanitized_name}'
        full_base_path = os.path.join(lora_folder, save_location_base)

        # Create folder structure if it doesn't exist
        os.makedirs(os.path.dirname(full_base_path), exist_ok=True)

        # Find next available version
        version_str = find_next_version(full_base_path)
        final_save_path = f"{full_base_path}_v{version_str}.safetensors"
        save_location_with_version = f"{save_location_base}_v{version_str}.safetensors"

        # Use vram_config to enable memory-efficient loading
        vram_config = {
            "offload_dtype": torch.bfloat16,  # Use bfloat16 for offloading to save VRAM
            "offload_device": "cpu",         # Offload to CPU when not in use
            "onload_dtype": torch.bfloat16,  # Use bfloat16 when loaded to GPU
            "onload_device": "cuda",
            "preparing_dtype": torch.bfloat16, # Use bfloat16 during preparation
            "preparing_device": "cpu",       # Prepare on CPU to save VRAM
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

        # Load only the models needed for Image2LoRA
        pipe = ZImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="SigLIP2-G384/model.safetensors", **vram_config),
                ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="DINOv3-7B/model.safetensors", **vram_config),
                ModelConfig(model_id="DiffSynth-Studio/Z-Image-i2L", origin_file_pattern="model.safetensors", **vram_config),
            ],
            tokenizer_config=None,
        )

        # Process images in batches
        num_images = len(images)
        
        encoder = ZImageUnit_Image2LoRAEncode()

        if batch_size >= num_images:
            # Process all at once
            with torch.no_grad():
                embs = encoder.process(pipe, image2lora_images=images)
        else:
            # Process in batches
            with torch.no_grad():
                embs = encoder.process(pipe, image2lora_images=images[:batch_size])

                for i in range(batch_size, num_images, batch_size):
                    batch_images = images[i:i + batch_size]
                    batch_embs = encoder.process(pipe, image2lora_images=batch_images)

                    # Merge embeddings
                    for key in embs:
                        if hasattr(embs[key], 'shape'):  # Tensor-like
                            embs[key] = torch.cat([embs[key], batch_embs[key]], dim=0)

                    del batch_embs
                    torch.cuda.empty_cache()

        # Generate LoRA
        decoder = CustomZImageUnit_Image2LoRADecode(lora_weights=lora_weights, normalized_strength=normalized_strength, reduce_size_factor=reduce_size_factor)
        lora = decoder.process(pipe, **embs)["lora"]

        # Save the LoRA file
        save_file(lora, final_save_path)

        # Return only the save_location (not full path)
        return save_location_with_version
else:
    # Stub implementations when DiffSynth-Studio is not available
    def process_images_to_lora(images_tensor, lora_name, batch_size, lora_weights=None, normalized_strength=1.0, reduce_size_factor=1):
        if lora_weights is None:
            lora_weights = []
        require_lora_filename(lora_name)
        raise ImportError("DiffSynth-Studio is not installed. Please install it following the instructions at https://github.com/modelscope/DiffSynth-Studio")

# Module exports for ComfyUI node
__all__ = ['process_images_to_lora', 'ZIMAGE_AVAILABLE', 'DIFF_SYNTH_AVAILABLE', 'DEPENDENCIES_AVAILABLE']