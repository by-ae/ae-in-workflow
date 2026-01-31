import pygame
import torch
import numpy as np
import math
import hashlib
import json
import os
import subprocess
import re
import platform
from pygame import surfarray

# pygame.init()  # Moved inside pose_editor function

# Cache file location (initialized lazily)
import os
CACHE_DIR = None
CACHE_FILE = None
WINDOW_CACHE_FILE = None

def get_cache_dir():
    """Get cache directory, initializing if needed"""
    global CACHE_DIR, CACHE_FILE, WINDOW_CACHE_FILE
    if CACHE_DIR is None:
        try:
            from folder_paths import get_user_directory
            CACHE_DIR = os.path.join(get_user_directory(), "ae-in-workflow", "pose_editor")
        except ImportError:
            # Fallback if folder_paths not available
            CACHE_DIR = os.path.expanduser("~/.comfyui_cache/ae-in-workflow/pose_editor")
        CACHE_FILE = os.path.join(CACHE_DIR, "cache.json")
        WINDOW_CACHE_FILE = os.path.join(CACHE_DIR, "window_position.json")

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

    return CACHE_DIR

# Undo/Redo constants
MAX_UNDO_STEPS = 5000  # Maximum undo steps to keep

# Common pose structure (same for all people)
# NOTE: Unsure if this is the correct "official order" as there was a lot of contradicting info on pose layouts
# From what I gather the colors, order, etc. are essentially varioua, but this one mostly aligns with cnet preprocessor colors
POSE_PAIRS = [
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13)
]

POSE_COLORS = [
    (0, 0, 152),    # (1, 0) neck → nose
    (54, 0, 152),   # (0, 14) nose → right eye
    (102, 0, 152),  # (14, 16) right eye → right ear
    (152, 0, 152),  # (0, 15) nose → left eye
    (152, 0, 102),  # (15, 17) left eye → left ear

    (152, 0, 0),   # (1, 2) neck → right shoulder
    (152, 102, 0),  # (2, 3) right shoulder → right elbow
    (152, 152, 0),  # (3, 4) right elbow → right wrist

    (152, 54, 0),   # (1, 5) neck → left shoulder
    (102, 152, 0),  # (5, 6) left shoulder → left elbow
    (54, 152, 0),   # (6, 7) left elbow → left wrist

    (0, 152, 0),    # (1, 8) neck → right hip
    (0, 152, 54),   # (8, 9) right hip → right knee
    (0, 152, 102),  # (9, 10) right knee → right ankle

    (0, 152, 152),  # (1, 11) neck → left hip
    (0, 102, 152),  # (11, 12) left hip → left knee
    (0, 54, 152),   # (12, 13) left knee → left ankle

    (255, 0, 85)
]

def get_child_keypoints(parent_idx):
    """Get all direct child keypoints of a given keypoint index"""
    children = []
    for parent, child in POSE_PAIRS:
        if parent == parent_idx:
            children.append(child)
    return children

def get_all_descendants(root_idx):
    """Get all descendants (children, grandchildren, etc.) of a keypoint recursively"""
    descendants = set()
    to_process = [root_idx]

    while to_process:
        current = to_process.pop()
        children = get_child_keypoints(current)
        for child in children:
            if child not in descendants:
                descendants.add(child)
                to_process.append(child)

    return descendants


def get_nearest_bone(mouse_pos, people_keypoints, people_valid_keypoints, person_idx):
    """Find the bone (pair of keypoints) closest to the mouse position for the given person"""
    mouse_x, mouse_y = mouse_pos
    current_keypoints = people_keypoints[person_idx]
    valid_keypoints = people_valid_keypoints[person_idx]

    min_distance = float('inf')
    nearest_bone = None

    # Check all valid bone pairs
    for i, j in POSE_PAIRS:
        if i in valid_keypoints and j in valid_keypoints:
            p1 = current_keypoints[i]
            p2 = current_keypoints[j]

            # Calculate distance from mouse to line segment
            # Using distance to line formula
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            if dx == 0 and dy == 0:
                # Points are the same, use distance to point
                distance = ((p1[0] - mouse_x)**2 + (p1[1] - mouse_y)**2)**0.5
            else:
                # Distance to line segment
                t = ((mouse_x - p1[0]) * dx + (mouse_y - p1[1]) * dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))  # Clamp to segment

                closest_x = p1[0] + t * dx
                closest_y = p1[1] + t * dy
                distance = ((closest_x - mouse_x)**2 + (closest_y - mouse_y)**2)**0.5

            if distance < min_distance:
                min_distance = distance
                nearest_bone = (i, j)

    return nearest_bone

def get_descendants_with_neck_special_case(root_idx):
    """Get descendants with special case for neck - only head chain, no arms/legs"""
    if root_idx == 1:  # Neck
        # For neck, only return head-related descendants (nose, eyes, ears)
        # Exclude shoulders (2, 5) and hips (8, 11) and their descendants
        descendants = get_all_descendants(root_idx)
        # Remove arm and leg chains
        arm_leg_keypoints = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}  # All arm and leg keypoints
        return descendants - arm_leg_keypoints
    else:
        return get_all_descendants(root_idx)

def flip_person_horizontal(person_idx, people_keypoints, people_valid_keypoints, swap_left_right=True):
    """Flip a person horizontally around their vertical center line"""
    keypoints = people_keypoints[person_idx]

    # Find the vertical center line (average of shoulder positions or neck)
    if 1 in people_valid_keypoints[person_idx]:  # neck
        center_x = keypoints[1][0]
    elif 2 in people_valid_keypoints[person_idx] and 5 in people_valid_keypoints[person_idx]:  # shoulders
        center_x = (keypoints[2][0] + keypoints[5][0]) / 2
    else:
        # Fallback: use average of all valid keypoints
        valid_points = [keypoints[j] for j in people_valid_keypoints[person_idx]]
        if valid_points:
            center_x = sum(p[0] for p in valid_points) / len(valid_points)
        else:
            return  # No valid keypoints to flip

    # Define left-right keypoint pairs to swap
    lr_pairs = [
        (2, 5),   # shoulders
        (3, 6),   # elbows
        (4, 7),   # wrists
        (8, 11),  # hips
        (9, 12),  # knees
        (10, 13), # ankles
        (14, 15), # eyes
        (16, 17), # ears
    ]

    # Flip coordinates horizontally
    for i in range(len(keypoints)):
        if i in people_valid_keypoints[person_idx]:
            # Reflect over center line
            keypoints[i][0] = 2 * center_x - keypoints[i][0]

    # Swap left-right keypoints if requested
    if swap_left_right:
        for left_idx, right_idx in lr_pairs:
            if (left_idx in people_valid_keypoints[person_idx] and
                right_idx in people_valid_keypoints[person_idx]):
                # Swap the keypoints
                keypoints[left_idx], keypoints[right_idx] = keypoints[right_idx], keypoints[left_idx]


def get_pose_hash(pose_data):
    """Generate a hash for the pose data to use as cache key"""
    pose_str = json.dumps(pose_data, sort_keys=True)
    return hashlib.md5(pose_str.encode()).hexdigest()

def get_window_position_fallback(window_title):
    """Get window position using platform-specific system tools"""
    system = platform.system()

    if system == 'Linux':
        # Linux/X11: use xwininfo
        try:
            result = subprocess.run(['xwininfo', '-name', window_title],
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                output = result.stdout
                abs_x_match = re.search(r'Absolute upper-left X:\s*(\d+)', output)
                abs_y_match = re.search(r'Absolute upper-left Y:\s*(\d+)', output)

                if abs_x_match and abs_y_match:
                    return int(abs_x_match.group(1)), int(abs_y_match.group(1))
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

    elif system == 'Windows':
        # Windows: try PowerShell with Win32 API
        try:
            ps_command = f'''
            $window = Get-Process | Where-Object {{$_.MainWindowTitle -eq "{window_title}"}} | Select-Object -First 1
            if ($window) {{
                $handle = $window.MainWindowHandle
                Add-Type @"
                    using System;
                    using System.Runtime.InteropServices;
                    public class Win32 {{
                        [DllImport("user32.dll")]
                        public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
                        [StructLayout(LayoutKind.Sequential)]
                        public struct RECT {{ public int Left; public int Top; public int Right; public int Bottom; }}
                    }}
"@
                $rect = New-Object Win32+RECT
                [Win32]::GetWindowRect($handle, [ref]$rect)
                Write-Output "$($rect.Left),$($rect.Top)"
            }}
            '''
            result = subprocess.run(['powershell', '-Command', ps_command],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

    elif system == 'Darwin':  # macOS
        # macOS: try osascript with System Events
        try:
            script = f'''
            tell application "System Events"
                set procList to (every process whose name is "Python")
                if (count of procList) > 0 then
                    set frontProc to item 1 of procList
                    tell frontProc
                        set windowList to (every window whose name contains "{window_title}")
                        if (count of windowList) > 0 then
                            set frontWindow to item 1 of windowList
                            set windowBounds to bounds of frontWindow
                            return windowBounds
                        end if
                    end tell
                end if
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                # osascript returns bounds as: left,top,right,bottom
                bounds = result.stdout.strip().split(',')
                if len(bounds) == 4:
                    return int(bounds[0]), int(bounds[1])
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

    return None

def create_t_pose():
    """Create a natural-looking T-pose template based on human proportions"""
    # Natural T-pose keypoints (x, y) coordinates
    # Based on typical human proportions: head ~1/8, torso ~1/3, etc.
    center_x = 400
    base_y = 180  # Started higher for more natural head position
    # TODO: extract a good default average human t-pose to use as a template
    # This one is more than fine for now though.

    t_pose_keypoints = [
        [center_x, base_y],      # 0: Nose
        [center_x, base_y + 25], # 1: Neck
        [center_x - 40, base_y + 30],  # 2: Right shoulder (slightly lower than neck)
        [center_x - 90, base_y + 45],  # 3: Right elbow (angled slightly down)
        [center_x - 140, base_y + 50], # 4: Right wrist
        [center_x + 40, base_y + 30],  # 5: Left shoulder
        [center_x + 90, base_y + 45],  # 6: Left elbow
        [center_x + 140, base_y + 50], # 7: Left wrist
        [center_x - 25, base_y + 120], # 8: Right hip
        [center_x - 25, base_y + 220], # 9: Right knee
        [center_x - 25, base_y + 320], # 10: Right ankle
        [center_x + 25, base_y + 120], # 11: Left hip
        [center_x + 25, base_y + 220], # 12: Left knee
        [center_x + 25, base_y + 320], # 13: Left ankle
        [center_x - 15, base_y - 15],  # 14: Right eye
        [center_x + 15, base_y - 15],  # 15: Left eye
        [center_x - 30, base_y - 10],  # 16: Right ear
        [center_x + 30, base_y - 10],  # 17: Left ear
    ]
    return t_pose_keypoints

def fix_missing_keypoints(person_idx, people_keypoints, people_valid_keypoints, people_valid_pairs, people_valid_colors, people_original_confidences):
    """Fix missing keypoints on a person using scaled T-pose proportions"""
    t_pose = create_t_pose()
    person_keypoints = people_keypoints[person_idx]
    valid_keypoints = people_valid_keypoints[person_idx]

    # Find neck position (keypoint 1)
    if 1 not in valid_keypoints:
        return 0  # Can't fix without neck reference

    neck_x, neck_y = person_keypoints[1]
    t_pose_neck_x, t_pose_neck_y = t_pose[1]

    # Calculate scale factor based on existing keypoints
    scale_factors = []
    for i in valid_keypoints:
        if i < len(t_pose) and i != 1:  # Skip neck itself
            # Distance from neck in person's pose
            person_dx = person_keypoints[i][0] - neck_x
            person_dy = person_keypoints[i][1] - neck_y
            person_dist = math.sqrt(person_dx*person_dx + person_dy*person_dy)

            # Distance from neck in T-pose
            t_pose_dx = t_pose[i][0] - t_pose_neck_x
            t_pose_dy = t_pose[i][1] - t_pose_neck_y
            t_pose_dist = math.sqrt(t_pose_dx*t_pose_dx + t_pose_dy*t_pose_dy)

            if t_pose_dist > 0:  # Avoid division by zero
                scale_factors.append(person_dist / t_pose_dist)

    # Use median scale factor to be robust against outliers
    if scale_factors:
        scale_factors.sort()
        scale_factor = scale_factors[len(scale_factors) // 2]  # median
    else:
        scale_factor = 1.0  # fallback if no valid measurements

    print(f"Person {person_idx}: using scale factor {scale_factor:.2f} (from {len(scale_factors)} reference points)")

    fixed_count = 0

    # For each keypoint that's missing
    for i in range(len(person_keypoints)):
        if i not in valid_keypoints and i < len(t_pose):
            # Calculate scaled position relative to neck in T-pose
            t_pose_keypoint_x, t_pose_keypoint_y = t_pose[i]
            relative_x = (t_pose_keypoint_x - t_pose_neck_x) * scale_factor
            relative_y = (t_pose_keypoint_y - t_pose_neck_y) * scale_factor

            # Apply the scaled relative position to the person's neck
            person_keypoints[i] = [neck_x + relative_x, neck_y + relative_y]

            # Add to valid keypoints
            valid_keypoints.append(i)
            # Update confidence to make this keypoint visible
            if person_idx < len(people_original_confidences) and i < len(people_original_confidences[person_idx]):
                people_original_confidences[person_idx][i] = 1.0
            fixed_count += 1

    # Re-filter pairs and colors for this person to include new keypoints
    if fixed_count > 0:
        valid_pairs = [(i, j) for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i in valid_keypoints and j in valid_keypoints]
        valid_colors = [col for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i in valid_keypoints and j in valid_keypoints]
        people_valid_pairs[person_idx] = valid_pairs
        people_valid_colors[person_idx] = valid_colors

    return fixed_count

def space_out_people(people_keypoints, people_valid_keypoints):
    """Space out people who are at the same center location"""
    if len(people_keypoints) <= 1:
        return people_keypoints

    # Calculate centers for each person
    centers = []
    for i, keypoints in enumerate(people_keypoints):
        valid_points = [keypoints[j] for j in people_valid_keypoints[i]]
        if valid_points:
            cx = sum(p[0] for p in valid_points) / len(valid_points)
            cy = sum(p[1] for p in valid_points) / len(valid_points)
            centers.append((cx, cy, i))  # (x, y, person_index)
        else:
            centers.append((0, 0, i))  # fallback

    # Group people by center location (with some tolerance)
    tolerance = 50.0  # pixels
    groups = {}
    for cx, cy, idx in centers:
        found_group = False
        for group_center in groups:
            if abs(cx - group_center[0]) < tolerance and abs(cy - group_center[1]) < tolerance:
                groups[group_center].append(idx)
                found_group = True
                break
        if not found_group:
            groups[(cx, cy)] = [idx]

    # Space out groups with multiple people
    spaced_keypoints = people_keypoints.copy()
    for group_center, person_indices in groups.items():
        if len(person_indices) > 1:
            # Calculate spacing
            spacing = 200.0  # pixels between people
            total_width = (len(person_indices) - 1) * spacing
            start_x = group_center[0] - total_width / 2

            # Check if we need to scale down to fit screen
            info = pygame.display.Info()
            screen_width = info.current_w
            scale_factor = 1.0
            if total_width > screen_width * 0.8:
                scale_factor = (screen_width * 0.8) / total_width

            # Only move the highest-indexed person (newest addition) in each group
            # Leave existing people at their positions
            max_idx = max(person_indices)
            for i, person_idx in enumerate(person_indices):
                if person_idx == max_idx:
                    # This is the newest person - move it to a spaced position
                    offset_x = (start_x + i * spacing * scale_factor) - group_center[0]

                    # Apply offset and scaling to all keypoints of this person
                    for j, point in enumerate(spaced_keypoints[person_idx]):
                        dx = point[0] - group_center[0]
                        dy = point[1] - group_center[1]
                        spaced_keypoints[person_idx][j][0] = group_center[0] + dx * scale_factor + offset_x
                        spaced_keypoints[person_idx][j][1] = group_center[1] + dy * scale_factor
                # Other people in the group stay at their current positions

    return spaced_keypoints

def get_closest_person(mouse_pos, people_keypoints, people_valid_keypoints):
    """Find the person with the closest visible keypoint to the mouse position"""
    if not people_keypoints:
        return 0

    mouse_x, mouse_y = mouse_pos
    closest_person = 0
    min_distance = float('inf')

    for i, keypoints in enumerate(people_keypoints):
        valid_keypoint_indices = people_valid_keypoints[i]
        for j in valid_keypoint_indices:
            px, py = keypoints[j]
            distance = (px - mouse_x) ** 2 + (py - mouse_y) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_person = i

    return closest_person

def draw_rotated_ellipse(surface, color, center, width, height, angle):
    """Draw a rotated ellipse using OpenPose-style approach"""
    # NOTE: I eyeballed this, it *seemed* like they were using rotated narrow ellipses
    # Seems to look and work fine so YOLO
    # Create a surface for the ellipse
    ellipse_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    # Draw ellipse on the surface (filled)
    pygame.draw.ellipse(ellipse_surface, color, (0, 0, width, height))
    # Rotate the surface
    rotated_surface = pygame.transform.rotate(ellipse_surface, angle)
    # Get the rect for positioning
    rect = rotated_surface.get_rect(center=center)
    # Blit to the main surface
    surface.blit(rotated_surface, rect)

def pose_editor(pose_data, padding=100, x=20, y=20, w=1024, h=1024, reset_cached_window_position=False):
    """
    Pose Editor - by-ae

    CONTROLS:
    - Left click: Select keypoint on closest person (any visible keypoint)
    - Left drag: Move selected keypoint
    - Ctrl + left drag: Move selected keypoint and all its children
    - Middle click + drag: Move selected person
    - Scroll: Zoom selected person
    - Ctrl + scroll: Rotate selected person
    - Shift + scroll: Rotate children around nearest keypoint (1.5x faster)
    - Ctrl + D: Duplicate selected person at their current position
    - Ctrl + X: Delete selected person (minimum 1 person required)
    - Ctrl + N: Add new person in T-pose template
    - Ctrl + F: Fix missing keypoints on selected person using T-pose proportions
    - Ctrl + R: Flip selected person horizontally (mirror pose, swaps left/right)
    - Ctrl + Shift + R: Flip selected person horizontally (turn around, no swap)
    - Ctrl + Z: Undo last action
    - Ctrl + Shift + Z / Ctrl + Y: Redo action
    - Ctrl + O: Reset to original input pose data
    - ESC: Exit and save all people

    Multiple people at same location are automatically spaced out.
    Selected person is highlighted with white outline.

    Returns: (tensor, pose_data) - tensor cropped to pose bounds with padding, pose_data for saving/loading
    """
    # Initialize pygame and font system
    pygame.init()
    pygame.font.init()
    # Store original input for reset functionality and consistent hashing
    if pose_data is None:
        # Create default T-pose when no pose data is provided
        t_pose_keypoints = create_t_pose()
        # Convert to pose_keypoints_2d format: [x1, y1, conf1, x2, y2, conf2, ...]
        pose_keypoints_2d = []
        for point in t_pose_keypoints:
            pose_keypoints_2d.extend([point[0], point[1], 1.0])  # Full confidence for T-pose
        print("Using default T-pose (no input pose data provided)")
        pose_data = [{"people": [{"pose_keypoints_2d": pose_keypoints_2d}]}]
    
    original_input = pose_data

    # Always use the hash of the original input for caching (not cached data)
    input_hash = get_pose_hash(original_input)
    print(f"Input hash: {input_hash}")

    # Read cache directly from file
    cached_data = None
    try:
        cache_dir = get_cache_dir()
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                file_cache = json.load(f)
                if input_hash in file_cache:
                    cached_data = file_cache[input_hash]
                    print("CACHE HIT - Found cached data in file")
                else:
                    print(f"CACHE MISS - Hash not in file (file has {len(file_cache)} entries)")
        else:
            print("CACHE MISS - No cache file exists")
    except Exception as e:
        print(f"Cache read error: {e}")

    if cached_data:
        print("Loading cached pose data as starting point...")
        print(f"Cached data type: {type(cached_data)}")
        if isinstance(cached_data, list) and len(cached_data) > 0:
            print(f"Cached canvas: {cached_data[0].get('canvas_width', '?')}x{cached_data[0].get('canvas_height', '?')}")
            print(f"Cached people: {len(cached_data[0].get('people', []))}")
            # Debug: check cached keypoints
            for person_idx, person in enumerate(cached_data[0].get('people', [])):
                pose = person.get('pose_keypoints_2d', [])
                print(f"Cached person {person_idx}: {len(pose)//3} keypoints")
                visible_count = 0
                for i in range(len(pose)//3):
                    x, y, conf = pose[3*i], pose[3*i+1], pose[3*i+2]
                    if conf > 0:
                        visible_count += 1
                        print(f"  Keypoint {i}: ({x:.1f}, {y:.1f}) conf={conf}")
                missing_count = (len(pose)//3) - visible_count
                print(f"  Visible: {visible_count}, Missing: {missing_count}")
        pose_data = cached_data  # Use cached data as starting point
        print("Using cached data as input for editing")
        # When using cached data, the cached confidences become the "original" confidences
    else:
        print("CACHE MISS - Starting with original input")

    try:
        people_data = pose_data[0]["people"]
    except Exception as e:
        print(f"Error parsing input data: {e}")
        return None, None

    # Initialize data for each person
    people_keypoints = []
    people_valid_keypoints = []
    people_original_confidences = []  # Store original confidence values

    for person in people_data:
        pose = person["pose_keypoints_2d"]
        n_points = len(pose) // 3
        keypoints = [[pose[3*i], pose[3*i+1]] for i in range(n_points)]
        confidences = [pose[3*i+2] for i in range(n_points)]  # Store original confidences
        # Valid keypoints must be both at non-zero position AND have confidence > 0
        valid_keypoints = [i for i in range(n_points) if (pose[3*i] != 0.0 or pose[3*i+1] != 0.0) and pose[3*i+2] > 0.0]
        people_keypoints.append(keypoints)
        people_valid_keypoints.append(valid_keypoints)
        people_original_confidences.append(confidences)

    # Store original positions before spacing (for duplication)
    original_people_keypoints = [keypoints.copy() for keypoints in people_keypoints]

    # Handle window position and size caching
    if reset_cached_window_position:
        print("Reset cached window position/size - using defaults")
        window_x, window_y, window_w, window_h = x, y, w, h
    else:
        # Try to load cached window position and size
        window_x, window_y, window_w, window_h = x, y, w, h  # defaults
        try:
            get_cache_dir()  # Ensure cache dir is initialized
            if os.path.exists(WINDOW_CACHE_FILE):
                with open(WINDOW_CACHE_FILE, 'r') as f:
                    cached_data = json.load(f)
                    if isinstance(cached_data, dict):
                        # Load cached values if they exist, otherwise use defaults
                        window_x = cached_data.get('x', x)
                        window_y = cached_data.get('y', y)
                        window_w = cached_data.get('w', w)
                        window_h = cached_data.get('h', h)
                        print(f"Loaded cached window: {window_x}, {window_y}, {window_w}x{window_h}")
                    else:
                        print("Invalid cached window data format")
        except Exception as e:
            print(f"Failed to load cached window data: {e}")

    # Handle people at same locations by spacing them out
    people_keypoints = space_out_people(people_keypoints, people_valid_keypoints)

    # Center all people on the pygame surface
    if people_keypoints:
        # Use the surface dimensions for centering (not screen dimensions)
        surface_center_x = window_w // 2
        surface_center_y = window_h // 2

        # Calculate center of all visible keypoints
        all_valid_points = []
        for person_idx in range(len(people_keypoints)):
            valid_points = [people_keypoints[person_idx][j] for j in people_valid_keypoints[person_idx]]
            all_valid_points.extend(valid_points)

        if all_valid_points:
            pose_center_x = sum(p[0] for p in all_valid_points) / len(all_valid_points)
            pose_center_y = sum(p[1] for p in all_valid_points) / len(all_valid_points)

            # Calculate offset to center the pose on the surface
            offset_x = surface_center_x - pose_center_x
            offset_y = surface_center_y - pose_center_y

            # Apply centering offset to all keypoints
            for person_keypoints in people_keypoints:
                for point in person_keypoints:
                    point[0] += offset_x
                    point[1] += offset_y

    # Use the global pose structure constants

    # Pre-filter pairs and colors for each person (they have the same structure)
    people_valid_pairs = []
    people_valid_colors = []
    for person_idx in range(len(people_keypoints)):
        valid_keypoints = people_valid_keypoints[person_idx]
        valid_pairs = [(i, j) for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i in valid_keypoints and j in valid_keypoints]
        valid_colors = [col for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i in valid_keypoints and j in valid_keypoints]
        people_valid_pairs.append(valid_pairs)
        people_valid_colors.append(valid_colors)

    # Find the currently focused window and match its cached position/size
    print(f"Creating window at ({window_x}, {window_y}) with size ({window_w}, {window_h})")
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
    screen = pygame.display.set_mode((window_w, window_h), pygame.RESIZABLE)
    pygame.display.set_caption("Pose Editor - ae")
    print(f"Window created, actual size: {pygame.display.get_window_size()}")

    # Initialize smaller font for on-screen display
    try:
        font = pygame.font.SysFont('monospace', 12)
    except:
        font = pygame.font.Font(None, 16)  # Fallback

    clock = pygame.time.Clock()
    running = True

    # Variables for multi-person editing
    selected_person = 0  # index of currently selected person
    selected_keypoint = None  # index of selected keypoint within selected person
    radius = 10
    thickness = 6

    # Variables for pose manipulation (per-person)
    middle_dragging = False
    last_mouse_pos = None
    dragging_with_children = False
    mouse_down_on_keypoint = None
    mouse_down_pos = None
    drag_state = None  # Will store {'original_positions': {...}, 'start_pos': (x,y)}

    # Undo/Redo system
    undo_buffer = []  # List of state snapshots
    undo_index = -1   # Current position in undo buffer (-1 means no undo history)
    continuous_operation_active = False  # Track if continuous operations are happening

    def save_undo_state():
        """Save current editor state to undo buffer"""
        nonlocal undo_buffer, undo_index

        # Create a deep copy of the current state
        state = {
            'people_keypoints': [[point.copy() for point in person] for person in people_keypoints],
            'people_valid_keypoints': [set(valid) for valid in people_valid_keypoints],  # Use sets for easier copying
            'people_original_confidences': [[conf for conf in person_confs] for person_confs in people_original_confidences],
            'selected_person': selected_person
        }

        # Remove any redo states (states after current undo_index)
        undo_buffer[:] = undo_buffer[:undo_index + 1]

        # Add new state
        undo_buffer.append(state)
        undo_index += 1

        # Limit buffer size
        if len(undo_buffer) > MAX_UNDO_STEPS:
            undo_buffer.pop(0)
            undo_index -= 1

    def restore_undo_state(state):
        """Restore editor state from undo buffer"""
        nonlocal people_keypoints, people_valid_keypoints, people_original_confidences, selected_person

        # Restore state
        people_keypoints = [[point.copy() for point in person] for person in state['people_keypoints']]
        people_valid_keypoints = [list(valid) for valid in state['people_valid_keypoints']]  # Convert back to lists
        people_original_confidences = [[conf for conf in person_confs] for person_confs in state['people_original_confidences']]
        selected_person = state['selected_person']

        # Rebuild dependent data structures
        people_valid_pairs[:] = []
        people_valid_colors[:] = []
        for person_idx in range(len(people_keypoints)):
            valid_keypoints = people_valid_keypoints[person_idx]
            valid_pairs = [(i, j) for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i in valid_keypoints and j in valid_keypoints]
            valid_colors = [col for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i in valid_keypoints and j in valid_keypoints]
            people_valid_pairs.append(valid_pairs)
            people_valid_colors.append(valid_colors)

    # Center all people on screen initially (but don't move them if they're already reasonably positioned)
    all_points = []
    for person_keypoints in people_keypoints:
        all_points.extend(person_keypoints)

    while running:
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Ctrl+D: Duplicate selected person at their current position
                    if selected_person < len(people_keypoints):
                        # Create a copy of the current keypoints for the selected person
                        duplicated_keypoints = [point.copy() for point in people_keypoints[selected_person]]
                        duplicated_valid_keypoints = people_valid_keypoints[selected_person].copy()

                        # Add the duplicated person to all data structures
                        original_people_keypoints.append([point.copy() for point in duplicated_keypoints])  # Current position becomes "original" for the duplicate
                        people_keypoints.append(duplicated_keypoints)
                        people_valid_keypoints.append(duplicated_valid_keypoints)
                        people_valid_pairs.append(people_valid_pairs[selected_person].copy())
                        people_valid_colors.append(people_valid_colors[selected_person].copy())
                        # Add confidence array for the duplicated person (copy from original)
                        people_original_confidences.append(people_original_confidences[selected_person].copy())

                                                                # Re-space people to handle the new duplicate
                        people_keypoints = space_out_people(people_keypoints, people_valid_keypoints)

                        # Select the new duplicate
                        selected_person = len(people_keypoints) - 1

                        print(f"Duplicated person at current position, now have {len(people_keypoints)} people")

                        # Save state after duplication
                        save_undo_state()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_z:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        # Ctrl+Shift+Z: Redo
                        if undo_index < len(undo_buffer) - 1:
                            undo_index += 1
                            restore_undo_state(undo_buffer[undo_index])
                            print(f"Redo: Restored state {undo_index + 1}/{len(undo_buffer)}")
                        else:
                            print("Redo: No more states to redo")
                    else:
                        # Ctrl+Z: Undo
                        if undo_index > 0:
                            undo_index -= 1
                            restore_undo_state(undo_buffer[undo_index])
                            print(f"Undo: Restored state {undo_index + 1}/{len(undo_buffer)}")
                        else:
                            print("Undo: No more states to undo")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_y:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Ctrl+Y: Redo
                    if undo_index < len(undo_buffer) - 1:
                        undo_index += 1
                        restore_undo_state(undo_buffer[undo_index])
                        print(f"Redo: Restored state {undo_index + 1}/{len(undo_buffer)}")
                    else:
                        print("Redo: No more states to redo")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Ctrl+O: Reset to original input pose data
                    print("Resetting to original input pose data...")
                    # Reinitialize with original input
                    try:
                        people_data = original_input[0]["people"]
                        # Reinitialize all data structures with original input
                        people_keypoints = []
                        people_valid_keypoints = []
                        people_original_confidences = []  # Reinitialize confidences too
                        for person in people_data:
                            pose = person["pose_keypoints_2d"]
                            n_points = len(pose) // 3
                            keypoints = [[pose[3*i], pose[3*i+1]] for i in range(n_points)]
                            confidences = [pose[3*i+2] for i in range(n_points)]  # Store original confidences
                            # Valid keypoints must be both at non-zero position AND have confidence > 0
                            valid_keypoints = [i for i in range(n_points) if (pose[3*i] != 0.0 or pose[3*i+1] != 0.0) and pose[3*i+2] > 0.0]
                            people_keypoints.append(keypoints)
                            people_valid_keypoints.append(valid_keypoints)
                            people_original_confidences.append(confidences)

                        # Re-initialize other data structures
                        people_valid_pairs = []
                        people_valid_colors = []
                        for person_idx in range(len(people_keypoints)):
                            valid_keypoints = people_valid_keypoints[person_idx]
                            valid_pairs = [(i, j) for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i in valid_keypoints and j in valid_keypoints]
                            valid_colors = [col for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i in valid_keypoints and j in valid_keypoints]
                            people_valid_pairs.append(valid_pairs)
                            people_valid_colors.append(valid_colors)

                        # Reset spacing
                        people_keypoints = space_out_people(people_keypoints, people_valid_keypoints)

                        # Center all people on the pygame surface (same as initial loading)
                        if people_keypoints:
                            # Use the surface dimensions for centering (not screen dimensions)
                            surface_center_x = window_w // 2
                            surface_center_y = window_h // 2

                            # Calculate center of all visible keypoints
                            all_valid_points = []
                            for person_idx in range(len(people_keypoints)):
                                valid_points = [people_keypoints[person_idx][j] for j in people_valid_keypoints[person_idx]]
                                all_valid_points.extend(valid_points)

                            if all_valid_points:
                                pose_center_x = sum(p[0] for p in all_valid_points) / len(all_valid_points)
                                pose_center_y = sum(p[1] for p in all_valid_points) / len(all_valid_points)

                                # Calculate offset to center the pose on the surface
                                offset_x = surface_center_x - pose_center_x
                                offset_y = surface_center_y - pose_center_y

                                # Apply centering offset to all keypoints
                                for person_keypoints_list in people_keypoints:
                                    for point in person_keypoints_list:
                                        point[0] += offset_x
                                        point[1] += offset_y

                        # Reset selection
                        selected_person = 0
                        selected_keypoint = None

                        # Clear undo buffer after full reset
                        undo_buffer.clear()
                        undo_index = -1

                        print("Reset complete - back to original input pose")
                    except Exception as e:
                        print(f"Failed to reset: {e}")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Ctrl+X: Delete selected person
                    if len(people_keypoints) > 1:  # Don't delete if only one person left
                        # Remove from all data structures
                        del original_people_keypoints[selected_person]
                        del people_keypoints[selected_person]
                        del people_valid_keypoints[selected_person]
                        del people_valid_pairs[selected_person]
                        del people_valid_colors[selected_person]
                        del people_original_confidences[selected_person]

                        # Adjust selected_person index
                        if selected_person >= len(people_keypoints):
                            selected_person = len(people_keypoints) - 1

                        # Re-space remaining people
                        people_keypoints = space_out_people(people_keypoints, people_valid_keypoints)

                        print(f"Deleted person, now have {len(people_keypoints)} people")

                        # Save state after deletion
                        save_undo_state()
                    else:
                        print("Cannot delete the last person")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Ctrl+N: Add new person in T-pose
                    t_pose_keypoints = create_t_pose()
                    t_pose_valid_keypoints = list(range(len(t_pose_keypoints)))  # All keypoints are valid

                    # Add to all data structures
                    original_people_keypoints.append([point.copy() for point in t_pose_keypoints])
                    people_keypoints.append(t_pose_keypoints)
                    people_valid_keypoints.append(t_pose_valid_keypoints)

                    # Generate pairs and colors for the new person (same as others)
                    n_points = len(t_pose_keypoints)
                    valid_pairs = [(i, j) for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i < n_points and j < n_points]
                    valid_colors = [col for (i, j), col in zip(POSE_PAIRS, POSE_COLORS) if i < n_points and j < n_points]

                    people_valid_pairs.append(valid_pairs)
                    people_valid_colors.append(valid_colors)
                    # Add confidence array for the new T-pose person (all keypoints are valid)
                    people_original_confidences.append([1.0] * len(t_pose_keypoints))

                    # Re-space all people including the new one
                    people_keypoints = space_out_people(people_keypoints, people_valid_keypoints)

                    # Select the new person
                    selected_person = len(people_keypoints) - 1

                    print(f"Added new person in T-pose, now have {len(people_keypoints)} people")

                    # Save state after adding new person
                    save_undo_state()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Ctrl+F: Fix missing keypoints on selected person using T-pose proportions
                    if selected_person < len(people_keypoints):
                        fixed_count = fix_missing_keypoints(selected_person, people_keypoints, people_valid_keypoints, people_valid_pairs, people_valid_colors, people_original_confidences)
                        if fixed_count > 0:
                            print(f"Fixed {fixed_count} missing keypoints on person {selected_person}")
                            # Save state after fixing keypoints
                            save_undo_state()
                    else:
                        print(f"No missing keypoints to fix on person {selected_person}")
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        # Ctrl+Shift+R: Flip horizontally without swapping keypoints (turn around)
                        if selected_person < len(people_keypoints):
                            flip_person_horizontal(selected_person, people_keypoints, people_valid_keypoints, swap_left_right=False)
                            print(f"Flipped person {selected_person} horizontally (turned around)")
                            save_undo_state()
                    else:
                        # Ctrl+R: Flip horizontally with left/right keypoint swapping (mirror pose)
                        if selected_person < len(people_keypoints):
                            flip_person_horizontal(selected_person, people_keypoints, people_valid_keypoints, swap_left_right=True)
                            print(f"Flipped person {selected_person} horizontally (mirrored pose)")
                            save_undo_state()
            elif event.type == pygame.KEYUP:
                # Save undo state when Ctrl key is released after continuous operations
                if (event.key == pygame.K_LCTRL or event.key == pygame.K_RCTRL) and continuous_operation_active:
                    save_undo_state()
                    continuous_operation_active = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button - select keypoint on closest person
                    keys = pygame.key.get_pressed()
                    selected_person = get_closest_person(mouse, people_keypoints, people_valid_keypoints)
                    current_keypoints = people_keypoints[selected_person]
                    n_points = len(current_keypoints)
                    selected_keypoint = min(range(n_points), key=lambda i: (current_keypoints[i][0] - mouse[0])**2 + (current_keypoints[i][1] - mouse[1])**2)
                    if (current_keypoints[selected_keypoint][0] - mouse[0])**2 + (current_keypoints[selected_keypoint][1] - mouse[1])**2 > 30**2:
                        selected_keypoint = None
                        mouse_down_on_keypoint = None
                    else:
                        mouse_down_on_keypoint = selected_keypoint
                        mouse_down_pos = mouse
                        # Check if Ctrl is held for hierarchical dragging
                        dragging_with_children = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
                        # Initialize drag state if dragging with children
                        if dragging_with_children:
                            drag_state = {
                                'start_pos': mouse,
                                'original_positions': {}
                            }
                            # Store original positions of all descendants
                            descendants = get_all_descendants(selected_keypoint)
                            descendants.add(selected_keypoint)
                            for idx in descendants:
                                if idx < len(people_keypoints[selected_person]):
                                    p = people_keypoints[selected_person][idx]
                                    drag_state['original_positions'][idx] = [p[0], p[1]]
                elif event.button == 2:  # Middle mouse button - start dragging selected person
                    selected_person = get_closest_person(mouse, people_keypoints, people_valid_keypoints)
                    middle_dragging = True
                    last_mouse_pos = mouse
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button up
                    # Save undo state after completing a drag operation
                    if mouse_down_on_keypoint is not None:
                        save_undo_state()

                    selected_keypoint = None
                    mouse_down_on_keypoint = None
                    mouse_down_pos = None
                    dragging_with_children = False
                    drag_state = None
                elif event.button == 2:  # Middle mouse button up
                    if middle_dragging:  # Only save if we were actually dragging
                        save_undo_state()
                    middle_dragging = False
                    last_mouse_pos = None
            elif event.type == pygame.MOUSEWHEEL:
                keys = pygame.key.get_pressed()
                selected_person = get_closest_person(mouse, people_keypoints, people_valid_keypoints)
                current_keypoints = people_keypoints[selected_person]
                n_points = len(current_keypoints)

                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:  # Shift+scroll = rotate around nearest keypoint
                    # Find the closest keypoint to the mouse position
                    n_points = len(current_keypoints)
                    closest_keypoint = min(range(n_points), key=lambda i: (current_keypoints[i][0] - mouse[0])**2 + (current_keypoints[i][1] - mouse[1])**2)

                    # Only proceed if the closest keypoint is reasonably close to the mouse
                    if (current_keypoints[closest_keypoint][0] - mouse[0])**2 + (current_keypoints[closest_keypoint][1] - mouse[1])**2 <= 50**2:
                        # Rotate a little faster, it was annoying with the scroll wheel
                        # Not everyone has a free scrolling wheel
                        rotation_delta = 5.0 if event.y > 0 else -5.0

                        # Use the closest keypoint as the rotation center
                        cx, cy = current_keypoints[closest_keypoint]

                        # Get descendants of the closest keypoint (children to rotate around it)
                        descendants = get_descendants_with_neck_special_case(closest_keypoint)
                        # Don't include the keypoint itself in rotation - it stays in place

                        cos_delta = math.cos(math.radians(rotation_delta))
                        sin_delta = math.sin(math.radians(rotation_delta))

                        # Rotate all descendants around the keypoint center
                        for idx in descendants:
                            if idx < len(current_keypoints):
                                p = current_keypoints[idx]
                                # Rotate point around the keypoint center
                                dx = p[0] - cx
                                dy = p[1] - cy
                                # Apply rotation matrix: [cos, -sin; sin, cos]
                                p[0] = cx + dx * cos_delta - dy * sin_delta
                                p[1] = cy + dx * sin_delta + dy * cos_delta

                        # Bone rotation is a discrete operation per scroll - save undo state immediately
                        save_undo_state()
                elif keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:  # Ctrl+scroll = rotate selected person
                    rotation_delta = 5.0 if event.y > 0 else -5.0

                    # Rotate selected person's keypoints around their center
                    valid_points = [current_keypoints[j] for j in people_valid_keypoints[selected_person]]
                    if valid_points:
                        cx = sum(p[0] for p in valid_points) / len(valid_points)
                        cy = sum(p[1] for p in valid_points) / len(valid_points)

                        cos_delta = math.cos(math.radians(rotation_delta))
                        sin_delta = math.sin(math.radians(rotation_delta))

                        for p in current_keypoints:
                            # Rotate each point around the center by the delta angle
                            dx = p[0] - cx
                            dy = p[1] - cy
                            # Apply rotation matrix: [cos, -sin; sin, cos]
                            p[0] = cx + dx * cos_delta - dy * sin_delta
                            p[1] = cy + dx * sin_delta + dy * cos_delta

                        continuous_operation_active = True
                else:  # Normal scroll = zoom selected person
                    factor = 1.05 if event.y > 0 else 0.95
                    valid_points = [current_keypoints[j] for j in people_valid_keypoints[selected_person]]
                    if valid_points:
                        cx = sum(p[0] for p in valid_points) / len(valid_points)
                        cy = sum(p[1] for p in valid_points) / len(valid_points)
                        for p in current_keypoints:
                            p[0] = cx + factor * (p[0] - cx)
                            p[1] = cy + factor * (p[1] - cy)

                        # Zoom is a discrete operation - save undo state immediately
                        save_undo_state()

        if selected_keypoint is not None:
            if dragging_with_children and drag_state is not None:
                # Ctrl+drag: move all descendants of the selected keypoint
                # Calculate movement delta from drag start
                dx = mouse[0] - drag_state['start_pos'][0]
                dy = mouse[1] - drag_state['start_pos'][1]

                # Move all descendants by the same delta from their original positions
                for idx, (orig_x, orig_y) in drag_state['original_positions'].items():
                    people_keypoints[selected_person][idx][0] = orig_x + dx
                    people_keypoints[selected_person][idx][1] = orig_y + dy
            else:
                # Normal drag: move just the selected keypoint
                people_keypoints[selected_person][selected_keypoint][0], people_keypoints[selected_person][selected_keypoint][1] = mouse

        # Handle middle mouse dragging for selected person translation
        if middle_dragging and last_mouse_pos is not None:
            dx = mouse[0] - last_mouse_pos[0]
            dy = mouse[1] - last_mouse_pos[1]
            if dx != 0 or dy != 0:
                # Translate selected person's keypoints
                for p in people_keypoints[selected_person]:
                    p[0] += dx
                    p[1] += dy
                last_mouse_pos = mouse

        screen.fill((0, 0, 0))

        # Draw all people
        for person_idx in range(len(people_keypoints)):
            current_keypoints = people_keypoints[person_idx]
            valid_pairs = people_valid_pairs[person_idx]
            valid_colors = people_valid_colors[person_idx]
            valid_keypoints = people_valid_keypoints[person_idx]

            # bones as rotated ellipses (OpenPose style - only draw valid connections)
            for (i, j), col in zip(valid_pairs, valid_colors):
                p1 = np.array(current_keypoints[i])
                p2 = np.array(current_keypoints[j])

                # Calculate center point (mY, mX in OpenPose notation)
                center = ((p1 + p2) / 2).astype(int)

                # Calculate distance and angle (matching OpenPose calculation)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx*dx + dy*dy)
                angle = -math.degrees(math.atan2(dy, dx))  # Negate for correct rotation direction

                # Draw rotated ellipse for bone (matching OpenPose style)
                ellipse_width = max(int(length), 1)
                ellipse_height = thickness  # stickwidth in OpenPose
                # Darken colors by 0.6 factor like OpenPose does
                darkened_col = tuple(int(c * 0.6) for c in col)
                draw_rotated_ellipse(screen, darkened_col, center, ellipse_width, ellipse_height, angle)

            # keypoints with colors matching their connecting bones
            keypoint_radius = thickness  # Same radius as the narrow side of bones

            # Create mapping of keypoints to their connecting bone colors
            keypoint_colors = {}
            for (i, j), col in zip(valid_pairs, valid_colors):
                if i not in keypoint_colors:
                    keypoint_colors[i] = []
                if j not in keypoint_colors:
                    keypoint_colors[j] = []
                keypoint_colors[i].append(col)
                keypoint_colors[j].append(col)

            # Draw keypoints in the bright colors of their connecting bones
            for i in valid_keypoints:
                if i in keypoint_colors:
                    p = current_keypoints[i]
                    # Draw the keypoint in each connecting bone color
                    for col in keypoint_colors[i]:
                        pygame.draw.circle(screen, col, tuple(map(int, p)), keypoint_radius)

            # Highlight selected person/keypoint
            if person_idx == selected_person:
                if selected_keypoint is not None:
                    pygame.draw.circle(screen, (255, 255, 0), tuple(map(int, current_keypoints[selected_keypoint])), radius + 4, 3)
                else:
                    # Highlight selected person by drawing a subtle outline around all keypoints
                    for i in valid_keypoints:
                        p = current_keypoints[i]
                        pygame.draw.circle(screen, (255, 255, 255), tuple(map(int, p)), radius + 2, 1)

        # Draw on-screen info with smaller, colored text
        y_offset = 8

        # Author (smaller, subtle)
        author_text = font.render("ae", True, (150, 150, 150))
        screen.blit(author_text, (8, y_offset))
        y_offset += 16

        # Keybinds with colors and better formatting
        keybind_items = [
            ("CONTROLS", (220, 220, 255), True),  # Header in light blue
            ("", (180, 180, 180), False),  # Spacer
            ("Left Click: Select keypoint", (180, 180, 180), False),
            ("Left Drag: Move keypoint", (180, 180, 180), False),
            ("Ctrl+Left Drag: Move with children", (200, 200, 100), False),
            ("", (180, 180, 180), False),  # Spacer
            ("Middle Click+Drag: Move person", (180, 180, 180), False),
            ("Scroll: Zoom person", (180, 180, 180), False),
            ("Ctrl+Scroll: Rotate person", (180, 180, 180), False),
            ("Shift+Scroll: Rotate children around nearest keypoint", (200, 200, 100), False),
            ("", (180, 180, 180), False),  # Spacer
            ("Ctrl+D: Duplicate person", (150, 200, 150), False),
            ("Ctrl+X: Delete person", (200, 150, 150), False),
            ("Ctrl+N: Add T-pose person", (150, 200, 150), False),
            ("Ctrl+F: Fix missing keypoints", (200, 200, 150), False),
            ("Ctrl+R: Mirror flip person", (200, 150, 200), False),
            ("Ctrl+Shift+R: Turn flip person", (200, 150, 200), False),
            ("Ctrl+Z: Undo", (150, 200, 200), False),
            ("Ctrl+Shift+Z/Ctrl+Y: Redo", (150, 200, 200), False),
            ("Ctrl+O: Reset to original", (200, 150, 150), False),
            ("", (180, 180, 180), False),  # Spacer
            ("ESC: Save & Exit", (255, 180, 180), False),
        ]

        for text_content, color, is_header in keybind_items:
            if is_header:
                text = font.render(text_content, True, color)
                # Make headers slightly larger
                text = pygame.transform.scale(text, (int(text.get_width() * 1.1), int(text.get_height() * 1.1)))
            else:
                text = font.render(text_content, True, color)
            screen.blit(text, (8, y_offset))
            y_offset += 13 if not is_header else 15

        pygame.display.flip()
        clock.tick(60)

    # Calculate bounds of all VALID keypoints with padding for final cropped output
    all_points = []
    for person_idx, person_keypoints in enumerate(people_keypoints):
        valid_points = [person_keypoints[j] for j in people_valid_keypoints[person_idx]]
        all_points.extend(valid_points)
        print(f"Person {person_idx}: {len(valid_points)} valid points")

    if not all_points:
        print("No valid keypoints found, returning empty result")
        # Return a minimal tensor and empty pose data
        empty_tensor = torch.zeros(1, 64, 64, 3)
        return empty_tensor, {"canvas_height": 64, "canvas_width": 64, "people": []}

    if not all_points:
        # Fallback if no points
        H, W = 800, 800
        surf = pygame.Surface((W, H))
        surf.fill((0, 0, 0))
    else:
        # Calculate bounds with symmetric padding
        # padding parameter passed to function
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        print(f"Final bounds: x={min_x:.1f} to {max_x:.1f}, y={min_y:.1f} to {max_y:.1f}")

        # Add symmetric padding to all sides
        padded_min_x = min_x - padding
        padded_max_x = max_x + padding
        padded_min_y = min_y - padding
        padded_max_y = max_y + padding

        # Calculate precise dimensions to fit the padded bounds exactly
        raw_W = max(1, math.ceil(padded_max_x - padded_min_x + 1))
        raw_H = max(1, math.ceil(padded_max_y - padded_min_y + 1))

        # Cap maximum size to prevent memory issues (4096 is a reasonable limit)
        MAX_SIZE = 4096
        W = min(max(raw_W, 64), MAX_SIZE)
        H = min(max(raw_H, 64), MAX_SIZE)

        print(f"Surface size: {W}x{H} (raw: {raw_W}x{raw_H})")

        # Create surface with calculated bounds
        try:
            surf = pygame.Surface((W, H))
            surf.fill((0, 0, 0))
        except Exception as e:
            print(f"Failed to create surface {W}x{H}: {e}")
            # Fallback to a smaller size
            W, H = 1024, 768
            surf = pygame.Surface((W, H))
            surf.fill((0, 0, 0))
            print(f"Fallback to {W}x{H}")

        # Map padded bounds directly to surface bounds (no centering)
        # This ensures exact padding without any centering artifacts
        offset_x = -padded_min_x
        offset_y = -padded_min_y

        # Draw all people without additional scaling (cropped to bounds)
        for person_idx in range(len(people_keypoints)):
            current_keypoints = people_keypoints[person_idx]
            valid_pairs = people_valid_pairs[person_idx]
            valid_colors = people_valid_colors[person_idx]
            valid_keypoints = people_valid_keypoints[person_idx]

            # Draw bones
            for (i, j), col in zip(valid_pairs, valid_colors):
                p1 = np.array([current_keypoints[i][0] + offset_x, current_keypoints[i][1] + offset_y])
                p2 = np.array([current_keypoints[j][0] + offset_x, current_keypoints[j][1] + offset_y])

                # Calculate center point (mY, mX in OpenPose notation)
                center = ((p1 + p2) / 2).astype(int)

                # Calculate distance and angle (matching OpenPose calculation)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx*dx + dy*dy)
                angle = -math.degrees(math.atan2(dy, dx))  # Negate for correct rotation direction

                # Draw rotated ellipse for bone (matching OpenPose style)
                ellipse_width = max(int(length), 1)
                ellipse_height = thickness  # stickwidth in OpenPose
                # Darken colors by 0.6 factor like OpenPose does
                darkened_col = tuple(int(c * 0.6) for c in col)
                draw_rotated_ellipse(surf, darkened_col, center, ellipse_width, ellipse_height, angle)

            # Draw keypoints with colors matching their connecting bones
            keypoint_radius = thickness/2.0  # Same radius as the narrow side of bones

            # Create mapping of keypoints to their connecting bone colors
            keypoint_colors = {}
            for (i, j), col in zip(valid_pairs, valid_colors):
                if i not in keypoint_colors:
                    keypoint_colors[i] = []
                if j not in keypoint_colors:
                    keypoint_colors[j] = []
                keypoint_colors[i].append(col)
                keypoint_colors[j].append(col)

            # Draw keypoints in the bright colors of their connecting bones
            for i in valid_keypoints:
                p = current_keypoints[i]
                sp = (p[0] + offset_x, p[1] + offset_y)

                if i in keypoint_colors and keypoint_colors[i]:
                    # Draw keypoint in colors of connecting bones
                    for col in keypoint_colors[i]:
                        pygame.draw.circle(surf, col, tuple(map(int, sp)), keypoint_radius)
                else:
                    # Draw isolated keypoints in white
                    pygame.draw.circle(surf, (255, 255, 255), tuple(map(int, sp)), keypoint_radius)

    # Convert pygame surface to tensor with correct orientation
    arr = surfarray.array3d(surf)
    # pygame gives (W, H, 3), we want (H, W, 3) for tensor
    arr = np.transpose(arr, (1, 0, 2))  # Convert to (H, W, 3)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3) uint8

    # Save window position and size to cache before quitting
    try:
        get_cache_dir()  # Ensure cache dir is initialized
        # Get current window size
        current_size = pygame.display.get_window_size()

        # Try to get window position using platform-specific tools
        current_pos = (x, y)  # fallback to function defaults
        pos_from_system = get_window_position_fallback("Pose Editor - ae")
        if pos_from_system:
            current_pos = pos_from_system
            print(f"Got position using system tools: {current_pos}")
        else:
            print(f"Could not get window position using system tools on {platform.system()}")

        window_data = {
            'x': current_pos[0],
            'y': current_pos[1],
            'w': current_size[0],
            'h': current_size[1]
        }
        print(f"Saving window to cache: {WINDOW_CACHE_FILE}")
        print(f"Window data: pos={current_pos}, size={current_size}")
        with open(WINDOW_CACHE_FILE, 'w') as f:
            json.dump(window_data, f, indent=2)
        print(f"Successfully saved window to cache: {current_pos[0]}, {current_pos[1]}, {current_size[0]}x{current_size[1]}")
    except Exception as e:
        print(f"Failed to save window data: {e}")
        import traceback
        traceback.print_exc()

    pygame.quit()

    # Reconstruct the pose data structure for saving/loading
    updated_pose_data = {"people": []}
    for person_idx, person_keypoints in enumerate(people_keypoints):
        pose_keypoints_2d = []
        original_confidences = people_original_confidences[person_idx]
        for keypoint_idx, point in enumerate(person_keypoints):
            # Use original confidence for keypoints that were originally missing/invalid
            # Use 1.0 for keypoints that were originally valid (visible and editable)
            original_conf = original_confidences[keypoint_idx] if keypoint_idx < len(original_confidences) else 0.0
            confidence = 1.0 if original_conf > 0.0 else 0.0
            pose_keypoints_2d.extend([point[0], point[1], confidence])
        updated_pose_data["people"].append({
            "pose_keypoints_2d": pose_keypoints_2d
        })

    # Return both the tensor and the updated pose data with actual canvas dimensions
    # Convert tensor to float in [0, 1] range
    # HACK:
    # Increase intensity of colors (excluding pure black) before normalization
    # This brings it up to match the levels of controlnet poses
    # We clip to 255 to avoid overflow
    arr_enhanced = np.where(arr == 0, 0, np.clip(arr * 2.0, 0, 255).astype(np.uint8))
    tensor = torch.from_numpy(arr_enhanced).unsqueeze(0).float() / 255.0 # float conversion for ComfyUI

    result_data = [{"canvas_height": H, "canvas_width": W, "people": updated_pose_data["people"]}]

    # Cache the result (use original input hash as key, store output pose data as value)
    print(f"SAVING TO CACHE - Hash: {input_hash}")
    print(f"Saving canvas: {result_data[0]['canvas_width']}x{result_data[0]['canvas_height']}")
    print(f"Saving people: {len(result_data[0]['people'])}")

    # Read existing cache, update it, and save back
    try:
        cache_dir = get_cache_dir()
        file_cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                file_cache = json.load(f)

        file_cache[input_hash] = result_data

        with open(CACHE_FILE, 'w') as f:
            json.dump(file_cache, f, indent=2)
        print(f"Successfully saved updated cache with {len(file_cache)} entries")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return tensor, result_data
