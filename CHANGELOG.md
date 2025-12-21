# Changelog

All notable changes to **ae-in-workflow** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2024-12-XX

### Added
- **Window Position & Size Caching**: Remembers exact window position and dimensions between sessions
- **Cross-platform Window Detection**: Native system tools for position detection (xwininfo/Linux, PowerShell/Windows, osascript/macOS)
- **Default T-Pose Support**: Automatically creates T-pose when no input pose data provided
- **Reset Window Position Parameter**: `reset_cached_window_position` boolean to reset cached window state

### Improved
- **Robust Font Initialization**: Fixed pygame font initialization issues
- **Better Centering Logic**: Surface-based centering instead of screen-based for consistent positioning
- **Enhanced Error Handling**: Graceful fallbacks for window position detection failures

### Fixed
- **Font Initialization**: Resolved "font not initialized" errors by moving pygame.init() inside pose_editor function
- **Window Positioning**: Fixed centering calculations to use actual surface dimensions

## [0.1.0] - 2024-12-XX

### Added
- **Pose Editor Node**: Interactive pose manipulation with full multi-person support
  - Individual keypoint editing with click and drag
  - Smart person selection (closest keypoint to mouse)
  - Multi-person scene handling with automatic spacing
  - Comprehensive keyboard controls (move, rotate, zoom, duplicate, delete, etc.)
  - Scale-aware keypoint fixing using T-pose proportions
  - Intelligent caching system for faster subsequent edits
  - Cross-platform focused window positioning
  - On-screen control instructions
- **Caching System**: Persistent pose data caching with MD5 hashing
- **Professional UI**: Clean on-screen display with keybinds and author info

### Technical Features
- Cross-platform compatibility (Windows, macOS, Linux)
- PyGame-based interactive interface
- OpenPose format support
- ComfyUI node integration
- Proper error handling and fallbacks
- Memory-efficient tensor operations

### Known Limitations
- Requires user interaction (not suitable for automated workflows)
- PyGame window management may have platform-specific quirks
- Cache files stored locally (not shared between installations)

---

## Development Notes

### Version History
- **v0.1.1**: Window caching, cross-platform improvements, default T-pose support
- **v0.1.0**: Initial release with Pose Editor node
- **Future**: Additional interactive nodes planned (mask editors, video processors, etc.)

### Contributing
Please report issues and feature requests on the [GitHub repository](https://github.com/ae-maker/ae-in-workflow).

---

**Legend:**
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities
