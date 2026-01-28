# Check for DiffSynth-Studio availability
try:
    from diffsynth.pipelines.z_image import (
        ZImagePipeline, ModelConfig,
        ZImageUnit_Image2LoRAEncode, ZImageUnit_Image2LoRADecode
    )
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

def process_images_to_lora(images_tensor, lora_name, batch_size):
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
    if batch_size >= num_images:
        # Process all at once
        with torch.no_grad():
            embs = ZImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images)
    else:
        # Process in batches
        with torch.no_grad():
            embs = ZImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images[:batch_size])

            for i in range(batch_size, num_images, batch_size):
                batch_images = images[i:i + batch_size]
                batch_embs = ZImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=batch_images)

                # Merge embeddings
                for key in embs:
                    if hasattr(embs[key], 'shape'):  # Tensor-like
                        embs[key] = torch.cat([embs[key], batch_embs[key]], dim=0)

                del batch_embs
                torch.cuda.empty_cache()

    # Generate LoRA
    lora = ZImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]

    # Save the LoRA file
    save_file(lora, final_save_path)

    # Return only the save_location (not full path)
    return save_location_with_version

# Module exports for ComfyUI node
__all__ = ['process_images_to_lora', 'ZIMAGE_AVAILABLE', 'DIFF_SYNTH_AVAILABLE', 'DEPENDENCIES_AVAILABLE']