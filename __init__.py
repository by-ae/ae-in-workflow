"""
ae-in-workflow - In-workflow Nodes for ComfyUI
==============================================

A collection of interactive nodes that provide heavy interaction and streaming capabilities for ComfyUI workflows.

Current nodes:
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

# Node registration
NODE_CLASS_MAPPINGS = {
    "PoseEditorAE": PoseEditorNodeAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseEditorAE": "Interactive Pose Editor (ae)",
}

# Web directory for any static assets (if needed in future)
WEB_DIRECTORY = "./web"
