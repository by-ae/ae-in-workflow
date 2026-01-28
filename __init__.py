"""
ae-in-workflow - In-workflow Nodes for ComfyUI
==============================================

A collection of interactive nodes that provide heavy interaction and streaming capabilities for ComfyUI workflows.

Current nodes:
- Z-Image - Images To LoRA: Convert image batches to LoRA using DiffSynth-Studio's Z-Image pipeline with memory-efficient processing
- Image Selector: Interactive image selection from folders with thumbnail grid view and batch processing
- Pose Editor (Interactive): Professional pose manipulation with multi-person support, hierarchical editing, undo/redo, and intelligent caching
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
from image_selector import image_selector

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
            }
        }

    RETURN_TYPES = (any_typ,)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_TOOLTIPS = (
        "Relative path to the generated LoRA file (compatible with LoraLoaderModelOnly)",
    )

    FUNCTION = "images_to_lora"
    CATEGORY = "ae-in-workflow"

    def images_to_lora(self, images, lora_name, batch_size=8, seed=0):
        """
        Convert images to LoRA using Z-Image pipeline

        Args:
            images: Tensor of images (B, H, W, 3) to convert to LoRA
            lora_name: Name for the LoRA dataset
            batch_size: Number of images to process simultaneously

        Returns:
            str: Relative path to the generated LoRA file
        """
        if not DIFF_SYNTH_AVAILABLE:
            raise ImportError("DiffSynth-Studio is not installed yet.\nSimply follow the instructions at https://github.com/modelscope/DiffSynth-Studio\nRead the github for more info: https://github.com/by-ae/ae-in-workflow")

        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("You need to run `pip install -r requirements.txt` to install the required dependencies.\nRead the github for more info: https://github.com/by-ae/ae-in-workflow")

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

            # Process the images
            result_path = process_images_to_lora(images, lora_name, batch_size)
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
                    "tooltip": "Path to folder containing images to select from"
                }),
            },
            "optional": {
                "target_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Target width for resized images (0 = auto-calculate from largest image)"
                }),
                "target_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Target height for resized images (0 = auto-calculate from largest image)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": "Change this value to force the node to run again with the same inputs"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", any_typ)
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "Selected images as batched tensor in ComfyUI IMAGE format",
        "Alpha masks for selected images (for images with transparency)"
    )

    FUNCTION = "select_images"
    CATEGORY = "ae-in-workflow"

    def select_images(self, folder_path, target_width=0, target_height=0, seed=0):
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
            # Call the image selector function
            # It returns: image_list, mask_list, image_batch, mask_batch
            _, _, image_batch, mask_batch = image_selector(
                folder_path,
                target_width if target_width > 0 else None,
                target_height if target_height > 0 else None
            )

            # Return in ComfyUI format
            return (image_batch, mask_batch)

        except Exception as e:
            print(f"Image Selector Error: {e}")
            import traceback
            traceback.print_exc()

            # Return empty tensors on error
            empty_image = torch.empty(0, 0, 0, 3)
            empty_mask = torch.empty(0, 0, 0)
            return (empty_image, empty_mask)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PoseEditorAE": PoseEditorNodeAE,
    "ZImageImagesToLoRAAE": ZImageImagesToLoRANodeAE,
    "ImageSelectorAE": ImageSelectorNodeAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseEditorAE": "Interactive Pose Editor (ae)",
    "ZImageImagesToLoRAAE": "Z-Image Images To LoRA (ae)",
    "ImageSelectorAE": "Image Selector (ae)",
}

# Web directory for any static assets (if needed in future)
WEB_DIRECTORY = "./web"
