# In-workflow Nodes for ComfyUI

[![GitHub](https://img.shields.io/badge/GitHub-by--ae-informational)](https://github.com/by-ae)
[![Author](https://img.shields.io/badge/Author-by--ae-blue)](https://github.com/by-ae)

A collection of interactive nodes that provide **heavy interaction and streaming capabilities** for ComfyUI workflows.

![Screenshot](assets/screenshot.png)

## 🎯 Features

**🚀 Super Easy Pose Editing** - Click, drag, boop and you're done. No complex workflows or external tools needed!

- **⚡ One-Click Editing**: Click any keypoint and drag to move it instantly
- **🎯 Smart Person Selection**: Automatically selects the person closest to your cursor
- **👥 Multi-Person Scenes**: Add, remove, and manage multiple people in one scene
- **🔄 Pose Manipulation**: Move, rotate, and scale entire poses or individual limbs
- **🦴 Limb Controls**: Move keypoints with or without their "children" (connected joints)
- **🔄 Flip & Turn**: Mirror poses horizontally or rotate them 180°
- **✨ Auto-Fix Missing Keypoints**: Estimates missing joints using T-pose proportions
- **📐 Smart Spacing**: Automatically prevents people from overlapping
- **🔄 Duplicate & Clone**: Copy poses for creating variations quickly
- **🎬 Animation Chains**: Link multiple nodes for frame-by-frame stop-motion style animation
- **💾 Smart Caching**: Remembers your edits for instant loading next time
- **🖥️ Window Memory**: Remembers window position and size between sessions
- **🌐 Cross-Platform**: Works identically on Windows, macOS, and Linux
- **🎨 T-Pose Templates**: Start fresh with clean T-poses when no input provided

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

## 🔧 Requirements

> **Note:** When you run `pip install -r requirements.txt` from within your ComfyUI virtual environment, these packages will be installed automatically.

```txt
pygame>=2.0.0
torch>=1.9.0
numpy>=1.19.0
Pillow>=8.0.0
screeninfo>=0.8.0  # Optional: improves window positioning
```

## 🎨 Nodes

### Interactive Pose Editor (ae)

**Category:** `ae-in-workflow`

An interactive pose editor that allows users to manipulate OpenPose keypoints with full control.

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

## 📝 Usage Examples

### Basic Pose Editing
1. Load pose data from OpenPose detection
2. Connect to the Pose Editor node
3. Click "Queue" to launch the interactive editor
4. Edit poses using mouse and keyboard controls
5. Close the editor to get the modified pose data

### Multi-person Scene Creation
1. Start with a single person pose (or no input for default T-pose)
2. Use Ctrl+N to add more T-pose people
3. Position and edit each person individually
4. Use Ctrl+D to duplicate poses for variations
5. Export the complete multi-person scene

### Iterative Refinement and Animation
1. Edit a pose and save
2. Next time you load the same pose data, it starts with your previous edits
3. Use Ctrl+Z to reset to the original if needed
4. You can also link multiple Pose Editor nodes in your workflow to chain distinct poses for rudimentary animation

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
