# In-workflow Nodes for ComfyUI

[![GitHub](https://img.shields.io/badge/GitHub-by--ae-informational)](https://github.com/by-ae)
[![Author](https://img.shields.io/badge/Author-by--ae-blue)](https://github.com/by-ae)

A collection of interactive nodes that provide **heavy interaction and streaming capabilities** for ComfyUI workflows.

![Screenshot](assets/screenshot.png)

## 🎯 Features

- **Interactive Pose Editor**: Full-featured pose manipulation with multi-person support
- **Real-time Editing**: Live visual feedback during pose manipulation
- **Multi-person Support**: Handle complex scenes with multiple people
- **Advanced Controls**: Comprehensive keyboard shortcuts for efficient editing
- **Smart Caching**: Fast loading of previously edited poses + window position/size memory
- **Default T-Pose**: Start editing from a clean T-pose when no input is provided
- **Cross-platform**: Works on Windows, macOS, and Linux with native window positioning

## 📦 Installation

1. Clone (or download and extract) this repository into your ComfyUI `custom_nodes` directory:
   ```
   ComfyUI/
   ├── custom_nodes/ <- clone here
   │   └── ae-in-workflow/
   │       ├── __init__.py
   │       ├── pose_editor.py
   │       ├── setup.py
   │       ├── requirements.txt
   │       ├── pyproject.toml
   │       ├── CHANGELOG.md
   │       ├── LICENSE
   │       ├── MANIFEST.in
   │       ├── README.md
   │       ├── assets/
   │       │   └── screenshot.png
   │       └── web/
   │           └── index.html
   ```

2. Restart ComfyUI

## 🎨 Nodes

### Pose Editor (Interactive)

**Category:** `ae-in-workflow`

An interactive pose editor that allows users to manipulate OpenPose keypoints with full control.

#### Features:
- **Individual Keypoint Editing**: Click and drag any keypoint
- **Person Selection**: Automatically selects the person closest to your mouse
- **Multi-person Support**: Handle scenes with multiple people
- **Pose Manipulation**: Move, rotate, and scale entire poses
- **Smart Spacing**: Automatically spaces overlapping people
- **Keypoint Fixing**: Automatically estimates missing keypoints using T-pose proportions
- **Duplication**: Clone poses for creating variations
- **Caching**: Remembers your edits for faster subsequent sessions

#### Controls:
```
Left Click: Select and drag keypoints
Left Drag: Move keypoint
Ctrl + Left Drag: Move keypoint with children
Middle Click + Drag: Move selected person
Scroll: Zoom selected person
Ctrl + Scroll: Rotate selected person
Shift + Scroll: Rotate children around nearest keypoint
Ctrl + D: Duplicate selected person
Ctrl + X: Delete selected person
Ctrl + N: Add new T-pose person
Ctrl + F: Fix missing keypoints
Ctrl + R: Flip horizontally (mirror pose)
Ctrl + Shift + R: Flip horizontally (turn around)
Ctrl + Z: Undo last action
Ctrl + Shift + Z / Ctrl + Y: Redo action
Ctrl + O: Reset to original input
ESC: Save & Exit
```

#### Inputs:
- **pose_data** (optional): OpenPose format pose data as JSON string (if not provided, starts with default T-pose)
- **padding** (optional): Padding around poses in output (default: 128)
- **seed** (optional): Change to force re-execution with same pose data (default: 0)
- **reset_cached_window_position** (optional): Reset window position/size to defaults (default: False)

#### Outputs:
- **image**: Rendered pose visualization image
- **edited_pose_data**: Modified pose data in OpenPose format as JSON string

## 🔧 Requirements

- ComfyUI
- PyGame
- PyTorch
- NumPy
- PIL (Pillow)
- screeninfo (optional, for monitor detection)

## 📝 Usage Examples

### Basic Pose Editing
1. Load pose data from OpenPose detection
2. Connect to Pose Editor node
3. Click "Queue" to launch the interactive editor
4. Edit poses using mouse and keyboard controls
5. Close the editor to get the modified pose data

### Multi-person Scene Creation
1. Start with single person pose (or no input for default T-pose)
2. Use Ctrl+N to add more T-pose people
3. Position and edit each person individually
4. Use Ctrl+D to duplicate poses for variations
5. Export the complete multi-person scene

### Iterative Refinement
1. Edit a pose and save
2. Next time you load the same pose data, it starts with your previous edits
3. Use Ctrl+Z to reset to original if needed

## 🎮 Advanced Features

### Intelligent Person Selection
The editor automatically selects the person whose keypoints are closest to your mouse cursor, making it easy to work with crowded scenes.

### Default T-Pose Creation
When no pose data is provided as input, the editor automatically creates a natural T-pose template for immediate editing. Perfect for creating poses from scratch.

### Scale-Aware Keypoint Fixing
When using Ctrl+F to fix missing keypoints, the editor analyzes the scale of existing keypoints and applies proportional corrections based on T-pose anatomy.

### Smart Caching
- Each unique pose input gets cached separately
- Subsequent edits of the same pose start with your last saved version
- Cache persists between ComfyUI sessions

### Window Position & Size Memory
- Remembers exact window position and size between sessions
- Cross-platform support using native system tools (xwininfo, PowerShell, osascript)
- Use `reset_cached_window_position=True` to reset to defaults if window gets lost

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for the ComfyUI community
- Inspired by the need for more interactive workflow tools
- Thanks to the OpenPose project for the pose format standard

## 📞 Support

If you encounter any issues or have questions:

1. Check the ComfyUI console for error messages
2. Ensure all requirements are installed
3. Try resetting the pose cache if edits aren't saving
4. File an issue on GitHub with detailed reproduction steps

---

**Made with ❤️ by-ae**
