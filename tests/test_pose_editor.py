"""
Tests for pose_editor module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np


def test_module_imports():
    """Test that pose_editor can be imported"""
    import pose_editor
    assert hasattr(pose_editor, 'pose_editor')
    assert hasattr(pose_editor, 'create_t_pose')
    assert hasattr(pose_editor, 'get_pose_hash')


def test_create_t_pose():
    """Test T-pose template creation"""
    from pose_editor import create_t_pose
    
    t_pose = create_t_pose()
    
    # Should have 18 keypoints (standard OpenPose format)
    assert len(t_pose) == 18
    
    # Each keypoint should have [x, y] coordinates
    for point in t_pose:
        assert len(point) == 2
        assert isinstance(point[0], (int, float))
        assert isinstance(point[1], (int, float))
    
    # Verify some expected positions (neck should be below nose)
    nose = t_pose[0]
    neck = t_pose[1]
    assert neck[1] > nose[1], "Neck should be below nose"
    
    # Left and right shoulders should be symmetric
    right_shoulder = t_pose[2]
    left_shoulder = t_pose[5]
    assert abs(right_shoulder[0] - neck[0]) == abs(left_shoulder[0] - neck[0]), "Shoulders should be symmetric"


def test_get_pose_hash():
    """Test pose data hashing for caching"""
    from pose_editor import get_pose_hash
    
    pose_data1 = [{"people": [{"pose_keypoints_2d": [100, 200, 1.0, 150, 250, 1.0]}]}]
    pose_data2 = [{"people": [{"pose_keypoints_2d": [100, 200, 1.0, 150, 250, 1.0]}]}]
    pose_data3 = [{"people": [{"pose_keypoints_2d": [100, 200, 1.0, 150, 251, 1.0]}]}]
    
    hash1 = get_pose_hash(pose_data1)
    hash2 = get_pose_hash(pose_data2)
    hash3 = get_pose_hash(pose_data3)
    
    # Same data should produce same hash
    assert hash1 == hash2
    
    # Different data should produce different hash
    assert hash1 != hash3
    
    # Hash should be a string
    assert isinstance(hash1, str)
    assert len(hash1) > 0


def test_get_child_keypoints():
    """Test getting child keypoints from skeleton structure"""
    from pose_editor import get_child_keypoints
    
    # Keypoint 1 (neck) should have multiple children
    children = get_child_keypoints(1)
    assert len(children) > 0
    assert 0 in children  # nose is child of neck
    
    # Keypoint 0 (nose) should have children (eyes)
    children = get_child_keypoints(0)
    assert 14 in children or 15 in children  # eyes are children of nose


def test_get_all_descendants():
    """Test recursive descendant finding"""
    from pose_editor import get_all_descendants
    
    # Neck (1) should have many descendants (entire body)
    descendants = get_all_descendants(1)
    assert len(descendants) > 5
    assert 0 in descendants  # nose
    assert 2 in descendants or 5 in descendants  # shoulders
    
    # Elbow should have fewer descendants (just wrist)
    descendants = get_all_descendants(3)  # right elbow
    assert 4 in descendants  # right wrist


def test_flip_person_horizontal():
    """Test horizontal flipping with left-right swap"""
    from pose_editor import flip_person_horizontal, create_t_pose
    
    # Create a simple pose
    people_keypoints = [create_t_pose()]
    people_valid_keypoints = [list(range(18))]
    
    # Store original positions (store x-coordinates)
    original_right_shoulder_x = people_keypoints[0][2][0]
    original_left_shoulder_x = people_keypoints[0][5][0]
    original_neck_x = people_keypoints[0][1][0]
    
    # Flip with left-right swap
    flip_person_horizontal(0, people_keypoints, people_valid_keypoints, swap_left_right=True)
    
    # After flip with swap:
    # 1. X-coordinates should be mirrored around center
    # 2. Left and right keypoints should be swapped
    new_right_shoulder_x = people_keypoints[0][2][0]
    new_left_shoulder_x = people_keypoints[0][5][0]
    
    # Right shoulder should now be where left shoulder was (mirrored)
    assert abs(new_right_shoulder_x - (2 * original_neck_x - original_left_shoulder_x)) < 1.0
    # Left shoulder should now be where right shoulder was (mirrored)
    assert abs(new_left_shoulder_x - (2 * original_neck_x - original_right_shoulder_x)) < 1.0


def test_pose_editor_validation_empty_people():
    """Test pose_editor handles empty people array"""
    import pose_editor
    
    # Empty people array
    invalid_pose = [{"people": []}]
    
    result = pose_editor.pose_editor(invalid_pose, padding=10, reset_cached_window_position=True)
    
    # Should return None for invalid input
    assert result[0] is None
    assert result[1] is None


def test_pose_editor_validation_missing_field():
    """Test pose_editor handles missing pose_keypoints_2d field"""
    import pose_editor
    
    # Missing pose_keypoints_2d field
    invalid_pose = [{"people": [{"invalid_field": []}]}]
    
    result = pose_editor.pose_editor(invalid_pose, padding=10, reset_cached_window_position=True)
    
    # Should return None for invalid input
    assert result[0] is None
    assert result[1] is None


def test_pose_editor_validation_malformed_keypoints():
    """Test pose_editor handles malformed keypoints array"""
    import pose_editor
    
    # Length not divisible by 3
    invalid_pose = [{"people": [{"pose_keypoints_2d": [100, 200, 1.0, 150]}]}]
    
    result = pose_editor.pose_editor(invalid_pose, padding=10, reset_cached_window_position=True)
    
    # Should return None for invalid input
    assert result[0] is None
    assert result[1] is None


def test_pose_editor_validation_wrong_type():
    """Test pose_editor handles wrong type for pose_keypoints_2d"""
    import pose_editor
    
    # pose_keypoints_2d is not a list
    invalid_pose = [{"people": [{"pose_keypoints_2d": "not a list"}]}]
    
    result = pose_editor.pose_editor(invalid_pose, padding=10, reset_cached_window_position=True)
    
    # Should return None for invalid input
    assert result[0] is None
    assert result[1] is None


def test_pose_node_rejects_malformed_pose():
    """The node raises a clear error instead of handing ComfyUI a None image."""
    from __init__ import PoseEditorNodeAE

    node = PoseEditorNodeAE()
    with pytest.raises(ValueError, match="Pose data is malformed"):
        node.edit_pose(POSE_KEYPOINT_optional=[{"people": []}])

    with pytest.raises(ValueError, match="pose_keypoints_2d"):
        node.edit_pose(POSE_KEYPOINT_optional={"people": [{"pose_keypoints_2d": [1, 2]}]})


def test_pose_editor_validation_empty_keypoints():
    """Test pose_editor handles empty keypoints array"""
    import pose_editor
    
    # Empty keypoints array
    invalid_pose = [{"people": [{"pose_keypoints_2d": []}]}]
    
    result = pose_editor.pose_editor(invalid_pose, padding=10, reset_cached_window_position=True)
    
    # Should return None for invalid input
    assert result[0] is None
    assert result[1] is None


def test_pose_editor_with_none_input():
    """Test pose_editor creates T-pose when input is None"""
    import pose_editor
    
    # This test would require pygame display, so we'll skip it in CI
    # But we can test the T-pose creation logic
    pytest.skip("Skipping interactive pygame test in CI")


def test_fix_missing_keypoints():
    """Test fixing missing keypoints using T-pose proportions"""
    from pose_editor import fix_missing_keypoints, create_t_pose
    
    # Create a pose with some missing keypoints
    t_pose = create_t_pose()
    people_keypoints = [t_pose]
    people_valid_keypoints = [[1, 2, 5]]  # Only neck and shoulders are valid
    people_valid_pairs = [[(1, 2), (1, 5)]]
    people_valid_colors = [[(255, 0, 0), (0, 255, 0)]]
    people_original_confidences = [[0.0] * 18]
    people_original_confidences[0][1] = 1.0
    people_original_confidences[0][2] = 1.0
    people_original_confidences[0][5] = 1.0
    
    fixed_count = fix_missing_keypoints(
        0, people_keypoints, people_valid_keypoints, 
        people_valid_pairs, people_valid_colors, people_original_confidences
    )
    
    # Should fix some missing keypoints
    assert fixed_count > 0
    assert len(people_valid_keypoints[0]) > 3


def test_space_out_people():
    """Test spacing out overlapping people"""
    from pose_editor import space_out_people, create_t_pose
    
    # Create two people at the same location
    t_pose1 = create_t_pose()
    t_pose2 = [point.copy() for point in create_t_pose()]  # Same position
    
    people_keypoints = [t_pose1, t_pose2]
    people_valid_keypoints = [list(range(18)), list(range(18))]
    
    # Space them out
    spaced_keypoints = space_out_people(people_keypoints, people_valid_keypoints)
    
    # Should return same number of people
    assert len(spaced_keypoints) == 2
    
    # The newest person (index 1) should have been moved
    # Calculate centers to verify spacing
    center1_x = sum(p[0] for p in spaced_keypoints[0]) / len(spaced_keypoints[0])
    center2_x = sum(p[0] for p in spaced_keypoints[1]) / len(spaced_keypoints[1])
    
    # Centers should be different after spacing
    assert center1_x != center2_x


def test_get_closest_person():
    """Test finding closest person to mouse position"""
    from pose_editor import get_closest_person, create_t_pose
    
    # Create two people at different locations
    t_pose1 = create_t_pose()
    t_pose2 = [[p[0] + 500, p[1]] for p in create_t_pose()]
    
    people_keypoints = [t_pose1, t_pose2]
    people_valid_keypoints = [list(range(18)), list(range(18))]
    
    # Mouse near first person
    mouse_pos = (400, 200)
    closest = get_closest_person(mouse_pos, people_keypoints, people_valid_keypoints)
    assert closest == 0
    
    # Mouse near second person
    mouse_pos = (900, 200)
    closest = get_closest_person(mouse_pos, people_keypoints, people_valid_keypoints)
    assert closest == 1


def test_get_nearest_bone():
    """Test finding nearest bone to mouse position"""
    from pose_editor import get_nearest_bone, create_t_pose
    
    t_pose = create_t_pose()
    people_keypoints = [t_pose]
    people_valid_keypoints = [list(range(18))]
    
    # Mouse near neck-nose bone
    nose = t_pose[0]
    neck = t_pose[1]
    mouse_pos = ((nose[0] + neck[0]) / 2, (nose[1] + neck[1]) / 2)
    
    nearest = get_nearest_bone(mouse_pos, people_keypoints, people_valid_keypoints, 0)
    
    # Should return a tuple of keypoint indices
    assert isinstance(nearest, tuple)
    assert len(nearest) == 2


def test_pose_pairs_structure():
    """Test that POSE_PAIRS structure is valid"""
    from pose_editor import POSE_PAIRS, POSE_COLORS
    
    # Should have bone connections
    assert len(POSE_PAIRS) > 0
    
    # Each pair should be a tuple of two indices
    for pair in POSE_PAIRS:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert isinstance(pair[0], int)
        assert isinstance(pair[1], int)
    
    # Should have matching colors
    assert len(POSE_COLORS) >= len(POSE_PAIRS)
    
    # Each color should be RGB tuple
    for color in POSE_COLORS:
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
