"""
ae-in-workflow - In-workflow Nodes for ComfyUI
==============================================

A collection of interactive nodes that provide heavy interaction and streaming capabilities for ComfyUI workflows.

Main nodes:
- Z-Image - Images To LoRA: Convert image batches to LoRA using DiffSynth-Studio's Z-Image pipeline with memory-efficient processing
- Image Selector: Interactive image selection from folders with thumbnail grid view and batch processing
- Pose Editor (Interactive): Professional pose manipulation with multi-person support, hierarchical editing, undo/redo, and intelligent caching

Utils:
- Interpolate Float List: Process and interpolate comma-separated float lists
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import hashlib
import json

# The famous `anytyp` trick for ComfyUI - Sorry I forget who did this originally
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_typ = AnyType("*")

# Add the current directory to sys.path so we can import pose_editor
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our pose editor
from pose_editor import pose_editor

# Import Z-Image functions
from zimage_i2l import process_images_to_lora, ZIMAGE_AVAILABLE, DIFF_SYNTH_AVAILABLE, DEPENDENCIES_AVAILABLE

# Import Image Selector functions
from image_selector import image_selector, hex_to_rgb

# Import model management and garbage collection
import comfy.model_management as model_management
import gc

# ComfyUI node class
class PoseEditorNodeAE:
    """
    Interactive Pose Editor Node - Professional pose manipulation for ComfyUI

    Advanced pose editing with hierarchical control, undo/redo, and multi-person support.

    Key Features:
    - Individual keypoint editing with click and drag
    - Hierarchical manipulation (move keypoints with their children)
    - Bone-level rotation (rotate limbs around joints)
    - Multi-person scene editing with automatic spacing
    - Complete undo/redo system
    - Intelligent caching for fast subsequent edits
    - Scale-aware keypoint fixing using anatomical proportions
    - Pose duplication, deletion, and transformation tools
    - Real-time visual feedback with professional UI
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "POSE_KEYPOINT_optional": (any_typ, {"tooltip": "OpenPose format pose data with keypoints"}),
                "padding": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Padding around poses in output image (pixels)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": "Change this value to force the node to run again with the same pose data"
                }),
                "reset_cached_window_position": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Usually False, but if you can't find the window set to True to reset.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", any_typ)
    RETURN_NAMES = ("image", "edited_pose_data")
    OUTPUT_TOOLTIPS = (
        "Rendered pose visualization image",
        "Modified pose data in OpenPose format with user edits applied"
    )

    FUNCTION = "edit_pose"
    CATEGORY = "ae-in-workflow"

    def edit_pose(self, POSE_KEYPOINT_optional=None, padding=128, seed=0, reset_cached_window_position=False):
        """
        Launch the interactive pose editor

        Args:
            POSE_KEYPOINT: (Optional) OpenPose format pose data with keypoints, if not provided, a default T-pose will be used
            padding: Padding around poses in output image (pixels)
            reset_cached_window_position: Whether to reset cached window position to defaults

        Returns:
            tuple: (image_tensor, edited_pose_data)
        """
        try:
            # Convert pose_data to the format expected by pose_editor
            if isinstance(POSE_KEYPOINT_optional, dict):
                input_data = [POSE_KEYPOINT_optional]
            else:
                input_data = POSE_KEYPOINT_optional

            # Call the pose editor
            image_tensor, edited_pose_data = pose_editor(input_data, padding=padding, reset_cached_window_position=reset_cached_window_position)

            return (image_tensor, edited_pose_data)

        except Exception as e:
            print(f"Pose Editor Error: {e}")
            import traceback
            traceback.print_exc()

            # Return a minimal fallback
            fallback_tensor = torch.zeros(1, 64, 64, 3)
            return (fallback_tensor, POSE_KEYPOINT_optional)




# ComfyUI node class for Z-Image - Images To LoRA conversion
class ZImageImagesToLoRANodeAE:
    """
    This node takes a batch of images and converts them into a LoRA (Low-Rank Adaptation)
    using the Z-Image i2L pipeline from DiffSynth-Studio.

    Powered by DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio

    The generated LoRA files are saved in: models/loras/Z-Image/ae/z-image_<name>vXXX.safetensors
    where XXX is an auto-incrementing version number.

    Key Features:
    - Memory-efficient processing with VRAM management
    - Batch processing for large image sets
    - Automatic version numbering to avoid overwrites
    - Returns relative path compatible with LoraLoaderModelOnly node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (any_typ, {"tooltip": "Batch of images as tensor (shape: B,H,W,3) to convert to LoRA"}),
                "lora_name": ("STRING", {
                    "default": "my_lora",
                    "tooltip": "Name for the LoRA dataset (will be sanitized for filename)"
                }),
            },
            "optional": {
                "batch_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of images to process simultaneously (higher = faster but more VRAM)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": "Change this value to force the node to run again with the same inputs"
                }),
                "unload_models_before_running": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload all models before running the node"
                }),
                "lora_weights": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated weights for each image (e.g., '1.0,2.0,0.5'). Leave empty for equal weighting."
                }),
                "normalize_saved_lora_strength_to": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.01,
                    "max": 5.00,
                    "step": 0.01,
                    "tooltip": "Identical behavior to setting LoRA Loader strength (use that for dialing in first).\nYour LoRA only works at 1.75 strength? No problem, set this to 1.75 and it will be the new 1.0"
                }),
                "reduce_size_factor": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Batch size for averaging LoRAs (higher = smaller final LoRA by averaging groups)"
                }),
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_TOOLTIPS = (
        "Relative path to the generated LoRA file (compatible with LoraLoaderModelOnly)",
    )

    FUNCTION = "images_to_lora"
    CATEGORY = "ae-in-workflow"

    def images_to_lora(self, images, lora_name, batch_size=8, lora_weights="", normalize_saved_lora_strength_to=1.0, reduce_size_factor=1, seed=0, unload_models_before_running=True):
        """
        Convert images to LoRA using Z-Image pipeline

        Args:
            images: Tensor of images (B, H, W, 3) to convert to LoRA
            lora_name: Name for the LoRA dataset
            batch_size: Number of images to process simultaneously
            lora_weights: List of weights for each image (for weighted averaging)
            normalize_saved_lora_strength_to: Target strength normalization for saved LoRA
            reduce_size_factor: Batch size for averaging LoRAs (reduces final size)

        Returns:
            str: Relative path to the generated LoRA file
        """
        if not DIFF_SYNTH_AVAILABLE:
            raise ImportError("DiffSynth-Studio is not installed yet.\nSimply follow the instructions at https://github.com/modelscope/DiffSynth-Studio\nRead the github for more info: https://github.com/by-ae/ae-in-workflow")

        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("You need to run `pip install -r requirements.txt` to install the required dependencies.\nRead the github for more info: https://github.com/by-ae/ae-in-workflow")

        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except:
            pass
        
        try:
            # Validate inputs
            if images is None:
                raise ValueError("Images input cannot be None")

            if not isinstance(images, torch.Tensor):
                raise ValueError("Images must be a torch tensor")

            if len(images.shape) != 4 or images.shape[-1] != 3:
                raise ValueError(f"Images tensor must have shape (B, H, W, 3), got {images.shape}")

            if images.shape[0] == 0:
                raise ValueError("Images tensor cannot be empty")

            # Parse lora weights from string
            weights_list = []
            if lora_weights.strip():
                try:
                    weights_list = [float(w.strip()) for w in lora_weights.split(',') if w.strip()]
                except ValueError:
                    print(f"Warning: Invalid lora_weights format '{lora_weights}', using equal weights")
                    weights_list = []

            # Process the images
            result_path = process_images_to_lora(images, lora_name, batch_size, weights_list, normalize_saved_lora_strength_to, reduce_size_factor)
            return (result_path,)

        except Exception as e:
            print(f"Z-Image - Images To LoRA Error: {e}")
            import traceback
            traceback.print_exc()
            raise e


# ComfyUI node class for Image Selector
class ImageSelectorNodeAE:
    """
    Image Selector Node - Interactive image selection from folder

    This node allows users to browse and select images from a folder using an interactive
    pygame-based UI. Selected images are automatically resized and batched into tensors
    suitable for use with other ComfyUI nodes.

    Features:
    - Browse images in folders with thumbnail grid view
    - Click to select/deselect images (maintains selection order)
    - Scroll through large image collections
    - Automatic resizing with aspect ratio preservation
    - Batch output in ComfyUI IMAGE format
    - Optional target dimensions for consistent sizing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to folder containing images to select from (or somewhere to start browsing)"
                }),
            },
            "optional": {
                "target_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Target width for resized images (0 = auto-calculate from largest image)"
                }),
                "target_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Target height for resized images (0 = auto-calculate from largest image)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": "Change this value to force the node to run again with the same inputs"
                }),
                "padding_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Hex color for image padding (e.g., #000000 for black, #FFFFFF for white)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", any_typ, "INT")
    RETURN_NAMES = ("images", "masks", "batch_count")
    OUTPUT_TOOLTIPS = (
        "Selected images as batched tensor in ComfyUI IMAGE format",
        "Alpha masks for selected images (for images with transparency)",
        "Number of images in the batch"
    )

    FUNCTION = "select_images"
    CATEGORY = "ae-in-workflow"

    def select_images(self, folder_path, target_width=0, target_height=0, seed=0, padding_color="#000000"):
        """
        Launch interactive image selector UI

        Args:
            folder_path: Path to folder containing images
            target_width: Target width for resizing (0 = auto)
            target_height: Target height for resizing (0 = auto)

        Returns:
            tuple: (image_batch_tensor, mask_batch_tensor)
        """
        try:
            # Validate inputs before proceeding
            if not folder_path or not isinstance(folder_path, str):
                raise ValueError(f"folder_path must be a non-empty string, got {type(folder_path).__name__}: {folder_path}")

            import os
            if not os.path.exists(folder_path):
                raise ValueError(f"folder_path does not exist: {folder_path}")
            if not os.path.isdir(folder_path):
                raise ValueError(f"folder_path is not a directory: {folder_path}")

            if target_width is not None and target_width > 0:
                if not isinstance(target_width, int):
                    raise ValueError(f"target_width must be an integer, got {type(target_width).__name__}: {target_width}")

            if target_height is not None and target_height > 0:
                if not isinstance(target_height, int):
                    raise ValueError(f"target_height must be an integer, got {type(target_height).__name__}: {target_height}")

            if padding_color:
                if not isinstance(padding_color, str):
                    raise ValueError(f"padding_color must be a string, got {type(padding_color).__name__}: {padding_color}")
                # Basic hex color validation (#RRGGBB or #RGB)
                import re
                if not re.match(r'^#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?$', padding_color):
                    raise ValueError(f"padding_color must be a valid hex color (e.g., #FF0000), got: {padding_color}")

            # Call the image selector function
            # It returns: image_list, mask_list, image_batch, mask_batch
            _, _, image_batch, mask_batch = image_selector(
                folder_path,
                target_width if target_width > 0 else None,
                target_height if target_height > 0 else None,
                padding_color
            )

            # Get batch count
            batch_count = image_batch.shape[0] if image_batch.numel() > 0 else 0

            # Return in ComfyUI format
            return (image_batch, mask_batch, batch_count)

        except Exception as e:
            # Re-raise the exception so ComfyUI displays it properly to the user
            raise e


# ComfyUI node class for Crop Image Batch to Mask Bounds
class CropImageBatchToMaskBoundsNodeAE:
    """
    Crop Image Batch to Mask Bounds - Crop images to mask boundaries with intelligent resizing

    This node crops images based on mask boundaries, with options for individual or combined cropping.
    Useful for preparing masked objects for further processing while maintaining consistent dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of images to crop (shape: B,H,W,C)"}),
                "masks": ("MASK", {"tooltip": "Batch of masks defining crop boundaries (shape: B,H,W)"}),
            },
            "optional": {
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask before finding bounds"
                }),
                "mode": (["mask_for_each", "sum_of_masks"], {
                    "default": "mask_for_each",
                    "tooltip": "mask_for_each: crop each to individual mask bounds, then resize then pad to match combined bounds.\nsum_of_masks: crop all to combined mask bounds"
                }),
                "padding_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Hex color for padding (e.g., #FF0000 for red)"
                }),
                "optional_padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Additional padding pixels to add around cropped area on each side"
                }),
                "optional_new_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Force output width to this size (0 = use combined mask bounds)"
                }),
                "optional_new_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Force output height to this size (0 = use combined mask bounds)"
                }),
                "color_fill_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill masked areas with padding color before processing"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "Cropped and resized images with consistent dimensions",
        "Corresponding masks cropped, resized, and padded to match images"
    )

    FUNCTION = "crop_to_mask_bounds"
    CATEGORY = "ae-in-workflow"

    def crop_to_mask_bounds(self, images, masks, invert_mask=False, mode="mask_for_each", padding_color="#000000", optional_padding=0, color_fill_mask=True, optional_new_width=0, optional_new_height=0):
        """
        Crop images to mask bounds with intelligent resizing

        Args:
            images: Tensor of images (B, H, W, C)
            masks: Tensor of masks (B, H, W)
            invert_mask: Whether to invert masks before cropping
            mode: "mask_for_each" or "sum_of_masks"
            padding_color: Hex color for padding
            optional_padding: Extra padding around cropped area
            color_fill_mask: Whether to fill mask areas with padding color

        Returns:
            tuple: (cropped_images, cropped_masks)
        """
        # Validate inputs
        if mode not in ["mask_for_each", "sum_of_masks"]:
            raise ValueError(f"mode must be 'mask_for_each' or 'sum_of_masks', got: {mode}")

        if not isinstance(optional_padding, int) or optional_padding < 0:
            raise ValueError(f"optional_padding must be a non-negative integer, got: {optional_padding}")

        if not isinstance(optional_new_width, int) or optional_new_width < 0:
            raise ValueError(f"optional_new_width must be a non-negative integer, got: {optional_new_width}")

        if not isinstance(optional_new_height, int) or optional_new_height < 0:
            raise ValueError(f"optional_new_height must be a non-negative integer, got: {optional_new_height}")

        # Validate padding_color format
        import re
        if not isinstance(padding_color, str) or not re.match(r'^#[0-9A-Fa-f]{3}([0-9A-Fa-f]{3})?$', padding_color):
            raise ValueError(f"padding_color must be a valid hex color (e.g., #FF0000), got: {padding_color}")

        # Convert hex color to RGB tuple (needed for pre-padding)
        try:
            padding_rgb = hex_to_rgb(padding_color)
        except ValueError:
            print(f"Invalid padding_color '{padding_color}', using black")
            padding_rgb = (0, 0, 0)

        # Pre-process: expand canvas to target dimensions if specified
        if optional_new_width > 0 or optional_new_height > 0:
            target_width = optional_new_width
            target_height = optional_new_height
            images, masks = self._expand_canvas(images, masks, padding_rgb, target_width, target_height)

        try:
            # padding_rgb already converted above

            # Process based on mode - use target dimensions as forced bounds if specified
            forced_bounds = (target_width, target_height) if 'target_width' in locals() and target_width > 0 else None

            if mode == "mask_for_each":
                return self._crop_mask_for_each(images, masks, invert_mask, padding_rgb, optional_padding, color_fill_mask, forced_bounds)
            elif mode == "sum_of_masks":
                return self._crop_sum_of_masks(images, masks, invert_mask, padding_rgb, optional_padding, color_fill_mask, forced_bounds)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as e:
            # Re-raise the exception so ComfyUI displays it properly to the user
            raise e

    def _crop_mask_for_each(self, images, masks, invert_mask, padding_rgb, optional_padding, color_fill_mask, forced_bounds):
        """Crop each image to its individual mask bounds, then resize all to match bounds"""
        import torch
        import numpy as np

        batch_size = images.shape[0]
        height, width = images.shape[1], images.shape[2]

        cropped_images = []
        cropped_masks = []
        all_crop_heights = []
        all_crop_widths = []

        # First pass: crop each image individually
        for i in range(batch_size):
            img_tensor = images[i]  # (H, W, C)
            mask_tensor = masks[i]  # (H, W)

            # Convert tensors to numpy for processing
            img_np = (img_tensor * 255).clamp(0, 255).byte().cpu().numpy()
            mask_np = mask_tensor.cpu().numpy()

            # Apply color_fill_mask preprocessing (fill opposite of what invert_mask uses)
            if color_fill_mask:
                if invert_mask:
                    # Fill the mask areas (foreground) with padding color
                    fill_mask = mask_np > 0.5
                    img_np[fill_mask] = padding_rgb
                else:
                    # Fill the inverted areas (background) with padding color
                    inverted_mask = 1.0 - mask_np
                    fill_mask = inverted_mask > 0.5
                    img_np[fill_mask] = padding_rgb

            # Use original mask for bounds finding (don't invert for bounds)
            bounds_mask = mask_np
            if invert_mask:
                bounds_mask = 1.0 - mask_np

            # Find mask bounds
            mask_binary = bounds_mask > 0.5
            if not mask_binary.any():
                # Empty mask - use full image
                min_y, max_y = 0, height
                min_x, max_x = 0, width
            else:
                rows = np.any(mask_binary, axis=1)
                cols = np.any(mask_binary, axis=0)
                min_y, max_y = np.where(rows)[0][[0, -1]]
                min_x, max_x = np.where(cols)[0][[0, -1]]

            # Add optional padding
            min_y = max(0, min_y - optional_padding)
            max_y = min(height, max_y + optional_padding)
            min_x = max(0, min_x - optional_padding)
            max_x = min(width, max_x + optional_padding)

            crop_height = max_y - min_y + 1
            crop_width = max_x - min_x + 1

            # Track dimensions
            all_crop_heights.append(crop_height)
            all_crop_widths.append(crop_width)

            # Crop image and original mask (not the inverted one)
            cropped_img = img_np[min_y:max_y, min_x:max_x]
            cropped_mask = mask_np[min_y:max_y, min_x:max_x]  # Always use original mask

            cropped_images.append((cropped_img, crop_width, crop_height))
            cropped_masks.append((cropped_mask, crop_width, crop_height))

        # Determine final canvas dimensions
        if forced_bounds:
            # Use forced bounds
            canvas_width, canvas_height = forced_bounds
        else:
            # Use combined bounds (existing behavior)
            canvas_width = max(all_crop_widths) if all_crop_widths else 1
            canvas_height = max(all_crop_heights) if all_crop_heights else 1

        # Second pass: scale each cropped image UP to fit within canvas dimensions (maintaining aspect ratio), then pad
        final_images = []
        final_masks = []

        for (cropped_img, orig_w, orig_h), (cropped_mask, _, _) in zip(cropped_images, cropped_masks):
            # Scale UP to touch canvas bounds while maintaining aspect ratio
            scale_w = canvas_width / orig_w
            scale_h = canvas_height / orig_h

            # Use the LARGER scale to ensure at least one dimension touches the edge
            scale = max(scale_w, scale_h)

            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            # Ensure we don't exceed canvas bounds
            if new_w > canvas_width or new_h > canvas_height:
                scale = min(canvas_width / orig_w, canvas_height / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

            # Ensure at least 1 pixel
            new_w = max(1, new_w)
            new_h = max(1, new_h)

            # Resize the cropped image and mask
            from PIL import Image
            pil_img = Image.fromarray(cropped_img)
            pil_mask = Image.fromarray((cropped_mask * 255).astype(np.uint8), mode='L')

            resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized_mask = pil_mask.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Create canvas with target dimensions
            canvas = np.full((canvas_height, canvas_width, 3), padding_rgb, dtype=np.uint8)
            mask_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

            # Center the resized content
            y_offset = (canvas_height - new_h) // 2
            x_offset = (canvas_width - new_w) // 2

            # Paste resized content
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = np.array(resized_img)
            mask_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = np.array(resized_mask).astype(np.float32) / 255.0

            # Convert back to tensors
            img_tensor = torch.from_numpy(canvas).float() / 255.0
            mask_tensor = torch.from_numpy(mask_canvas)

            final_images.append(img_tensor)
            final_masks.append(mask_tensor)

        # Batch tensors
        image_batch = torch.stack(final_images, dim=0)  # (B, H, W, C)
        mask_batch = torch.stack(final_masks, dim=0)    # (B, H, W)

        return image_batch, mask_batch

    def _crop_sum_of_masks(self, images, masks, invert_mask, padding_rgb, optional_padding, color_fill_mask, forced_bounds):
        """Crop all images to the combined bounds of all masks"""
        import torch
        import numpy as np

        batch_size = images.shape[0]
        height, width = images.shape[1], images.shape[2]

        # Apply color_fill_mask preprocessing to all images
        processed_images = []
        processed_masks = []

        for i in range(batch_size):
            img_tensor = images[i]
            mask_tensor = masks[i]

            img_np = (img_tensor * 255).clamp(0, 255).byte().cpu().numpy()
            mask_np = mask_tensor.cpu().numpy()

            # Apply color_fill_mask preprocessing (fill opposite of what invert_mask uses)
            if color_fill_mask:
                if invert_mask:
                    # Fill the mask areas (foreground) with padding color
                    fill_mask = mask_np > 0.5
                    img_np[fill_mask] = padding_rgb
                else:
                    # Fill the inverted areas (background) with padding color
                    inverted_mask = 1.0 - mask_np
                    fill_mask = inverted_mask > 0.5
                    img_np[fill_mask] = padding_rgb

            processed_images.append(img_np)
            processed_masks.append(mask_np)

        # Find combined bounds across all masks
        global_min_y = height
        global_max_y = 0
        global_min_x = width
        global_max_x = 0

        for i in range(batch_size):
            mask_np = processed_masks[i]
            bounds_mask = mask_np
            if invert_mask:
                bounds_mask = 1.0 - mask_np

            mask_binary = bounds_mask > 0.5
            if mask_binary.any():
                rows = np.any(mask_binary, axis=1)
                cols = np.any(mask_binary, axis=0)
                min_y, max_y = np.where(rows)[0][[0, -1]]
                min_x, max_x = np.where(cols)[0][[0, -1]]

                global_min_y = min(global_min_y, min_y)
                global_max_y = max(global_max_y, max_y)
                global_min_x = min(global_min_x, min_x)
                global_max_x = max(global_max_x, max_x)

        # If no valid masks found, use full bounds
        if global_min_y >= global_max_y:
            global_min_y, global_max_y = 0, height
            global_min_x, global_max_x = 0, width

        # Add optional padding
        global_min_y = max(0, global_min_y - optional_padding)
        global_max_y = min(height, global_max_y + optional_padding)
        global_min_x = max(0, global_min_x - optional_padding)
        global_max_x = min(width, global_max_x + optional_padding)

        # Use forced bounds or calculated bounds
        if forced_bounds:
            crop_height, crop_width = forced_bounds
            # Use full image bounds for cropping
            global_min_y, global_max_y = 0, height
            global_min_x, global_max_x = 0, width
        else:
            crop_height = global_max_y - global_min_y
            crop_width = global_max_x - global_min_x

        # Crop all processed images and original masks to these bounds
        cropped_images = []
        cropped_masks = []

        for i in range(batch_size):
            img_np = processed_images[i]
            mask_np = processed_masks[i]  # Always use original mask for output

            # Crop
            cropped_img = img_np[global_min_y:global_max_y, global_min_x:global_max_x]
            cropped_mask = mask_np[global_min_y:global_max_y, global_min_x:global_max_x]

            # Convert back to tensors
            img_tensor = torch.from_numpy(cropped_img).float() / 255.0
            mask_tensor = torch.from_numpy(cropped_mask)

            cropped_images.append(img_tensor)
            cropped_masks.append(mask_tensor)

        # Batch tensors
        image_batch = torch.stack(cropped_images, dim=0)  # (B, H, W, C)
        mask_batch = torch.stack(cropped_masks, dim=0)    # (B, H, W)

        return image_batch, mask_batch

    def _expand_canvas(self, images, masks, padding_rgb, target_width, target_height):
        """Expand canvas to at least target dimensions by padding"""
        import torch
        import numpy as np

        batch_size = images.shape[0]
        current_height, current_width = images.shape[1], images.shape[2]

        # Expand dimensions if needed
        expand_width = max(target_width, current_width)
        expand_height = max(target_height, current_height)

        # If no expansion needed, return as-is
        if expand_width == current_width and expand_height == current_height:
            return images, masks

        padded_images = []
        padded_masks = []

        for i in range(batch_size):
            img_tensor = images[i]
            img_np = (img_tensor * 255).clamp(0, 255).byte().cpu().numpy()

            # Create expanded canvas
            expanded_img = np.full((expand_height, expand_width, 3), padding_rgb, dtype=np.uint8)

            # Copy original content
            expanded_img[:current_height, :current_width] = img_np

            # Same for mask (pad with 0)
            mask_tensor = masks[i]
            mask_np = mask_tensor.cpu().numpy()

            expanded_mask = np.zeros((expand_height, expand_width), dtype=np.float32)
            expanded_mask[:current_height, :current_width] = mask_np

            # Convert back to tensors
            img_tensor = torch.from_numpy(expanded_img).float() / 255.0
            mask_tensor = torch.from_numpy(expanded_mask)

            padded_images.append(img_tensor)
            padded_masks.append(mask_tensor)

        # Batch tensors
        image_batch = torch.stack(padded_images, dim=0)
        mask_batch = torch.stack(padded_masks, dim=0)

        return image_batch, mask_batch



# ComfyUI node class for Float List Interpolation
class InterpolateFloatListNodeAE:
    """
    Interpolate Float List - Process and interpolate lists of float values

    This node takes a comma-separated list of floats and either interpolates them
    across a specified count or trims/pads the list to match the count.

    Useful for creating smooth transitions between values or ensuring consistent
    list lengths for batch processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("STRING", {
                    "default": "3,2.5,2,1,0.5",
                    "multiline": True,
                    "tooltip": "Comma-separated list of float values (can use newlines, whitespace will be removed)"
                }),
            },
            "optional": {
                "count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Target count for interpolation/trimming (0 = return original list)"
                }),
                "interpolate_across_count": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Yes",
                    "label_off": "No pad right with 1's",
                    "tooltip": "If Yes: interpolate values across count. If No: trim/pad with 1s to match count"
                }),
            }
        }

    RETURN_TYPES = (any_typ, "STRING")
    RETURN_NAMES = ("float_list", "csv_string")
    OUTPUT_TOOLTIPS = (
        "List of interpolated/processed float values",
        "Comma-separated string of the float values"
    )

    FUNCTION = "interpolate_floats"
    CATEGORY = "ae-in-workflow"

    def interpolate_floats(self, floats, count=0, interpolate_across_count=True):
        """
        Process and interpolate float list

        Args:
            floats: Comma-separated string of float values
            count: Target count for processing
            interpolate_across_count: Whether to interpolate or trim/pad

        Returns:
            tuple: (list of processed float values, comma-separated string of values)
        """
        try:
            # Process the input string
            # Replace newlines with commas
            processed = floats.replace('\n', ',')
            # Remove all whitespace
            processed = ''.join(processed.split())
            # Parse into float list, filtering out empty strings
            values = [float(n) for n in processed.split(",") if n.strip() != ""]

            # If count <= 0, return original list
            if count <= 0:
                rounded_values = [round(v, 6) for v in values]
                csv_string = ','.join(str(v) for v in rounded_values)
                return (rounded_values, csv_string)

            if interpolate_across_count:
                # Interpolate across count items
                if len(values) <= 1:
                    # Not enough values to interpolate, return repeated first value
                    rounded_values = [round(values[0], 6)] * count if values else [1.0] * count
                    csv_string = ','.join(str(v) for v in rounded_values)
                    return (rounded_values, csv_string)
                else:
                    # Linear interpolation between values
                    result = []
                    for i in range(count):
                        # Calculate position in original array
                        pos = (i / max(1, count - 1)) * (len(values) - 1)
                        lower_idx = int(pos)
                        upper_idx = min(lower_idx + 1, len(values) - 1)
                        fraction = pos - lower_idx

                        # Interpolate between lower and upper values
                        interpolated_value = values[lower_idx] + (values[upper_idx] - values[lower_idx]) * fraction
                        result.append(round(interpolated_value, 6))

                    csv_string = ','.join(str(v) for v in result)
                    return (result, csv_string)
            else:
                # Trim/pad with 1s to match count
                if len(values) >= count:
                    # Trim to count
                    rounded_values = [round(v, 6) for v in values[:count]]
                    csv_string = ','.join(str(v) for v in rounded_values)
                    return (rounded_values, csv_string)
                else:
                    # Pad with 1s
                    result = [round(v, 6) for v in values]  # Copy the list with rounding
                    result.extend([1.0] * (count - len(values)))
                    csv_string = ','.join(str(v) for v in result)
                    return (result, csv_string)

        except Exception as e:
            print(f"Interpolate Float List Error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty list and empty string on error
            return ([], "")


# Node registration
NODE_CLASS_MAPPINGS = {
    "PoseEditorAE": PoseEditorNodeAE,
    "ZImageImagesToLoRAAE": ZImageImagesToLoRANodeAE,
    "ImageSelectorAE": ImageSelectorNodeAE,
    "CropImageBatchToMaskBoundsAE": CropImageBatchToMaskBoundsNodeAE,
    "InterpolateFloatListAE": InterpolateFloatListNodeAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseEditorAE": "Interactive Pose Editor (ae)",
    "ZImageImagesToLoRAAE": "Z-Image Images To LoRA (ae)",
    "ImageSelectorAE": "Image Selector (ae)",
    "CropImageBatchToMaskBoundsAE": "Crop Image Batch to Mask Bounds (ae)",
    "InterpolateFloatListAE": "Interpolate Float List (ae)",
}

# Web directory for any static assets (if needed in future)
WEB_DIRECTORY = "./web"
