# Check for DiffSynth-Studio availability
from typing import Any


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
    import os
    import re
    import folder_paths
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# Overall availability
ZIMAGE_AVAILABLE = DIFF_SYNTH_AVAILABLE and DEPENDENCIES_AVAILABLE

def merge_lora_weighted_average(loras, weights):
    """Average LoRAs with weights instead of concatenating - memory-efficient version"""
    if not loras:
        return {}
    
    merged = {}
    keys = [k for k in loras[0].keys() if ".lora_A." in k]
    
    # Precompute sum of weights once (scalar)
    sum_weights = sum(weights)
    if sum_weights <= 0:
        sum_weights = len(weights)  # fallback to uniform
    
    for key in keys:
        # NOTE: Fixed - use torch.zeros_like (correct API)
        first_A = loras[0][key]
        merged[key] = torch.zeros_like(first_A)          # same device, dtype, shape
        lora_B_key = key.replace(".lora_A.", ".lora_B.")
        first_B = loras[0][lora_B_key]
        merged[lora_B_key] = torch.zeros_like(first_B)
        
        for i, lora in enumerate(loras):
            w = weights[i]
            if w != 0:
                scale = w / sum_weights
                merged[key] += lora[key] * scale
                merged[lora_B_key] += lora[lora_B_key] * scale
    
    return merged


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
        if self.reduce_size_factor <= 1:
            return loras, weights
        
        # SMART SORTING: Group similar weights together
        sorted_pairs = sorted(zip(weights, loras), key=lambda x: x[0], reverse=True)
        sorted_weights, sorted_loras = zip[tuple[Any, ...]](*sorted_pairs)
        
        batch_size = self.reduce_size_factor
        num_loras = len(sorted_loras)
        
        batched_loras = []
        batched_weights = []
        
        for i in range(0, num_loras, batch_size):
            batch_end = min(i + batch_size, num_loras)
            batch_loras = sorted_loras[i:batch_end]
            batch_weights = list(sorted_weights[i:batch_end])
            
            if len(batch_loras) == 1:
                batched_loras.append(batch_loras[0])
                batched_weights.append(batch_weights[0])
            else:
                averaged_lora = merge_lora_weighted_average(batch_loras, batch_weights)
                batched_loras.append(averaged_lora)
                # Preserve total batch strength (sum, not avg) for consistency
                batched_weights.append(sum(batch_weights))
        
        print(f"Smart size reduction: {num_loras} → {len(batched_loras)} LoRAs (factor: {self.reduce_size_factor})")
        return batched_loras, batched_weights

def sanitize_lora_name(name):
    # Convert to lowercase and replace spaces/special chars with underscores
    name = re.sub(r'[^a-zA-Z0-9\s]', '', name).lower()
    name = re.sub(r'\s+', '_', name)
    return name

def find_next_version(base_path):
    # Find the next available version number (000, 001, 002, etc.)
    version = 0
    while True:
        version_str = f"{version:03d}"
        test_path = f"{base_path}_v{version_str}.safetensors"
        if not os.path.exists(test_path):
            return version_str
        version += 1

def process_images_to_lora(images_tensor, lora_name, batch_size, lora_weights=[], normalized_strength=1.0, reduce_size_factor=1):
    # Convert tensor to PIL images (assuming shape N, H, W, 3)
    images = []
    for i in range(images_tensor.shape[0]):
        # Convert tensor to PIL Image
        img_tensor = images_tensor[i]  # Shape: (H, W, 3)
        img_array = (img_tensor * 255).clamp(0, 255).byte().cpu().numpy()
        img = Image.fromarray(img_array)
        images.append(img)

    # Sanitize lora name
    sanitized_name = sanitize_lora_name(lora_name)

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
    # TODO: make this configurable perhaps, though it is already so fast.
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
    
    # Possible bug fix: Reuse the same encoder
    # I believe this was causing it to only use the last batch.
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

# Module exports for ComfyUI node
__all__ = ['process_images_to_lora', 'ZIMAGE_AVAILABLE', 'DIFF_SYNTH_AVAILABLE', 'DEPENDENCIES_AVAILABLE']