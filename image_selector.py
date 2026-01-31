import os
import pygame
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import resize
import math
import time
import json
import threading

def load_images_from_folder(folder_path):
    """Load all image files from folder"""
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    images = []
    paths = []

    if not os.path.exists(folder_path):
        return images, paths

    for file in os.listdir(folder_path):
        if file.lower().endswith(supported_formats):
            path = os.path.join(folder_path, file)
            try:
                img = Image.open(path)
                images.append(img)
                paths.append(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue

    return images, paths

def create_thumbnail(img, size):
    """Create thumbnail maintaining aspect ratio"""
    img_ratio = img.width / img.height
    thumb_ratio = size[0] / size[1]

    if img_ratio > thumb_ratio:
        # Image is wider than thumbnail area
        new_width = size[0]
        new_height = int(size[0] / img_ratio)
    else:
        # Image is taller than thumbnail area
        new_height = size[1]
        new_width = int(size[1] * img_ratio)

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Cache management
def get_cache_dir():
    """Get cache directory using folder_paths API"""
    try:
        import folder_paths
        return os.path.join(folder_paths.get_user_directory(), "ae-in-workflow", "image_selector")
    except ImportError:
        # Fallback
        return os.path.expanduser("~/.comfyui_cache/ae-in-workflow/image_selector")

def load_sort_settings():
    """Load sort settings from cache"""
    cache_dir = get_cache_dir()
    cache_file = os.path.join(cache_dir, "settings.json")

    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('sort_mode_index', 0), data.get('sort_ascending', True)
    except Exception as e:
        print(f"Error loading sort settings: {e}")

    return 2, False  # Default: modified date, descending

def save_sort_settings(sort_mode_index, sort_ascending):
    """Save sort settings to cache"""
    cache_dir = get_cache_dir()
    cache_file = os.path.join(cache_dir, "settings.json")

    try:
        os.makedirs(cache_dir, exist_ok=True)
        data = {
            'sort_mode_index': sort_mode_index,
            'sort_ascending': sort_ascending
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving sort settings: {e}")

def render_text_with_outline(font, text, text_color, outline_color=(0, 0, 0), outline_width=1):
    """Render text with black outline for better readability"""
    # Render the main text
    main_text = font.render(text, True, text_color)

    # Create a surface large enough for the text plus outline
    width = main_text.get_width() + outline_width * 2
    height = main_text.get_height() + outline_width * 2
    surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # Draw outline by rendering text in black at offset positions
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue  # Skip center position
            outline_text = font.render(text, True, outline_color)
            surface.blit(outline_text, (dx + outline_width, dy + outline_width))

    # Draw the main text on top
    surface.blit(main_text, (outline_width, outline_width))

    return surface

def load_folder_async(folder_path, sort_mode_index, sort_ascending, sort_modes, result_callback):
    """Load folder contents asynchronously"""
    def load_task():
        try:
            # Load images from folder
            _, paths = load_images_from_folder(folder_path)
            # Sort them
            paths = sort_image_paths(paths, sort_modes[sort_mode_index], sort_ascending)
            # Call back with results
            result_callback(paths)
        except Exception as e:
            print(f"Error loading folder {folder_path}: {e}")
            result_callback([])  # Return empty list on error

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()
    return thread

def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def build_folder_tree(root_path, current_path):
    """Build a hierarchical list of folders and files with back navigation"""
    tree_items = []

    try:
        # Add ".." for going up one directory (unless we're at root)
        parent_path = os.path.dirname(root_path)
        if parent_path != root_path:  # Not at filesystem root
            tree_items.append(("⬆️  ..", parent_path, True))

        # Add current directory
        tree_items.append(("📁 " + os.path.basename(root_path), root_path, True))

        # Add subdirectories
        for item in sorted(os.listdir(root_path)):
            full_path = os.path.join(root_path, item)
            if os.path.isdir(full_path):
                # Highlight current path
                prefix = "  📁 "
                if full_path == current_path and current_path != root_path:
                    prefix = "▶️ 📁 "
                tree_items.append((prefix + item, full_path, True))
    except:
        pass

    return tree_items

def sort_image_paths(paths, sort_mode, ascending=True):
    """Sort image paths by specified criteria without loading images"""
    if not paths:
        return paths

    # Create pairs of (path, sort_key)
    pairs = []

    for path in paths:
        try:
            if sort_mode == 'name':
                sort_key = os.path.basename(path).lower()
            elif sort_mode == 'size':
                sort_key = os.path.getsize(path)
            elif sort_mode == 'modified':
                sort_key = os.path.getmtime(path)
            elif sort_mode == 'created':
                sort_key = os.path.getctime(path)
            elif sort_mode == 'dimensions':
                # For dimensions, we need to load the image temporarily
                try:
                    with Image.open(path) as img:
                        sort_key = img.width * img.height  # Total pixels
                except:
                    sort_key = 0  # Fallback for unreadable images
            else:
                sort_key = os.path.basename(path).lower()  # Default to name
        except:
            sort_key = os.path.basename(path).lower()  # Fallback to name on error

        pairs.append((path, sort_key))

    # Sort the pairs
    reverse = not ascending
    if sort_mode in ['size', 'modified', 'created', 'dimensions']:
        # Numeric sorting
        pairs.sort(key=lambda x: x[1], reverse=reverse)
    else:
        # String sorting
        pairs.sort(key=lambda x: x[1], reverse=reverse)

    # Return sorted paths
    return [pair[0] for pair in pairs]

def sort_images(images, paths, sort_mode, ascending=True):
    """Sort images and paths by specified criteria (legacy function for compatibility)"""
    if not images or not paths:
        return images, paths

    # Create pairs of (image, path, sort_key)
    pairs = []

    for img, path in zip(images, paths):
        try:
            if sort_mode == 'name':
                sort_key = os.path.basename(path).lower()
            elif sort_mode == 'size':
                sort_key = os.path.getsize(path)
            elif sort_mode == 'modified':
                sort_key = os.path.getmtime(path)
            elif sort_mode == 'created':
                sort_key = os.path.getctime(path)
            elif sort_mode == 'dimensions':
                sort_key = img.width * img.height  # Total pixels
            else:
                sort_key = os.path.basename(path).lower()  # Default to name
        except:
            sort_key = os.path.basename(path).lower()  # Fallback to name on error

        pairs.append((img, path, sort_key))

    # Sort the pairs
    reverse = not ascending
    if sort_mode in ['size', 'modified', 'created', 'dimensions']:
        # Numeric sorting
        pairs.sort(key=lambda x: x[2], reverse=reverse)
    else:
        # String sorting
        pairs.sort(key=lambda x: x[2], reverse=reverse)

    # Unpack back to separate lists
    sorted_images = [pair[0] for pair in pairs]
    sorted_paths = [pair[1] for pair in pairs]

    return sorted_images, sorted_paths

def pil_to_tensor(img):
    """Convert PIL image to torch tensor (1,H,W,3)"""
    if img.mode == 'RGBA':
        # Handle transparency
        img_rgb = img.convert('RGB')
        mask = np.array(img)[:, :, 3]  # Alpha channel
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)  # (1,H,W)
    else:
        img_rgb = img
        # Create all-white mask
        mask_tensor = torch.ones(1, img.height, img.width)

    # Convert to tensor (H,W,C) format
    img_array = np.array(img_rgb)
    tensor = torch.from_numpy(img_array).float() / 255.0
    tensor = tensor.unsqueeze(0)  # (1,H,W,3)

    return tensor, mask_tensor

class LazyImageLoader:
    """Lazy loader for images to handle large folders efficiently"""
    def __init__(self, paths, thumbnail_size=(200, 200), cache_size=100):
        self.paths = paths
        self.thumbnail_size = thumbnail_size
        self.cache_size = cache_size
        self.cache = {}  # path -> (thumbnail_surface, last_access_time)
        self.access_counter = 0
        self.loading_queue = set()  # paths currently being loaded

    def get_thumbnail(self, path):
        """Get thumbnail for path, loading if necessary"""
        if path in self.cache:
            # Update access time
            self.cache[path] = (self.cache[path][0], self.access_counter)
            self.access_counter += 1
            return self.cache[path][0]

        # Check if already in loading queue
        if path in self.loading_queue:
            return None  # Still loading

        # Start loading (limit concurrent loads to prevent choppiness)
        if len(self.loading_queue) < 1:  # Max 1 concurrent load
            self.loading_queue.add(path)
            try:
                img = Image.open(path)
                # Resize more efficiently
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                # Create square thumbnail by padding
                thumb = Image.new('RGB', self.thumbnail_size, (0, 0, 0))
                x = (self.thumbnail_size[0] - img.width) // 2
                y = (self.thumbnail_size[1] - img.height) // 2
                thumb.paste(img, (x, y))

                # Convert PIL to pygame surface
                thumb_surface = pygame.image.fromstring(thumb.tobytes(), thumb.size, 'RGB')

                # Add to cache
                self.cache[path] = (thumb_surface, self.access_counter)
                self.access_counter += 1

                # Evict old items if cache is full
                if len(self.cache) > self.cache_size:
                    # Find oldest accessed item
                    oldest_path = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                    del self.cache[oldest_path]

            except Exception as e:
                # Create error placeholder
                thumb_surface = pygame.Surface(self.thumbnail_size)
                thumb_surface.fill((100, 100, 100))  # Gray background
                self.cache[path] = (thumb_surface, self.access_counter)
                self.access_counter += 1
            finally:
                self.loading_queue.discard(path)

        return self.cache.get(path, (None, 0))[0] if path in self.cache else None

    def preload_visible(self, visible_paths):
        """Preload thumbnails for currently visible images"""
        for path in visible_paths:
            if path not in self.cache and path not in self.loading_queue:
                # Start loading (but don't wait)
                self.loading_queue.add(path)
                # We'll load it when get_thumbnail is called

def create_image_selector_ui(paths, folder_path, padding_color="#000000", target_width_param=None, target_height_param=None, thumbnail_size=(150, 150)):
    """Create pygame UI for image selection with lazy loading"""
    pygame.init()

    # Screen setup - add space for folder tree
    tree_width = 250
    screen_width = 1200 + tree_width
    screen_height = 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Selector - Click to select, ESC to finish")

    # Colors
    BG_COLOR = (30, 30, 30)
    THUMB_BG = (50, 50, 50)
    SELECTED_COLOR = (100, 150, 255)
    BORDER_COLOR = (70, 70, 70)
    TEXT_COLOR = (220, 220, 220)
    TREE_BG = (40, 40, 40)
    TREE_SELECTED = (80, 80, 80)
    LOADING_COLOR = (80, 80, 120)  # Purple-ish for loading

    # Font
    font = pygame.font.SysFont('Arial', 16)

    # Grid setup - adjust for tree panel
    cols = 4
    padding = 20
    thumb_width, thumb_height = thumbnail_size
    cell_width = thumb_width + padding * 2
    cell_height = thumb_height + padding * 2 + 40  # Extra space for filename

    # Tree panel setup
    tree_panel_x = 0
    tree_panel_y = 0
    thumbnail_panel_x = tree_width
    thumbnail_panel_y = 0

    # Tree navigation state
    current_folder = os.path.abspath(folder_path)
    folder_tree = build_folder_tree(current_folder, current_folder)
    tree_scroll_offset = 0
    max_tree_scroll = max(0, len(folder_tree) * 25 - (screen_height - 100))

    # Sorting state
    sort_modes = ['name', 'size', 'modified', 'created', 'dimensions']
    # Load cached sort settings
    sort_mode_index, sort_ascending = load_sort_settings()

    # Sort paths initially
    paths = sort_image_paths(paths, sort_modes[sort_mode_index], sort_ascending)

    # Initialize lazy loader instead of loading all thumbnails
    lazy_loader = LazyImageLoader(paths, thumbnail_size)

    # Loading state for async folder changes
    loading_folder = False
    loading_thread = None

    # Selection state - track order of selection
    selected_indices = []  # Will store indices in order they were selected
    selected_set = set()   # For quick lookup of what's selected
    scroll_offset = 0
    max_scroll = max(0, ((len(paths) + cols - 1) // cols) * cell_height - screen_height + 100)

    running = True
    clock = pygame.time.Clock()
    select_all = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    running = False
                elif event.key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Ctrl+A to select/deselect all
                    select_all = not select_all
                    if select_all:
                        selected_indices = list(range(len(paths)))
                        selected_set = set(range(len(paths)))
                    else:
                        selected_indices = []
                        selected_set = set()
                elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Ctrl+S to cycle sort mode
                    sort_mode_index = (sort_mode_index + 1) % len(sort_modes)
                    save_sort_settings(sort_mode_index, sort_ascending)
                    paths = sort_image_paths(paths, sort_modes[sort_mode_index], sort_ascending)
                    # Clear cache and create new lazy loader with sorted paths
                    lazy_loader = LazyImageLoader(paths, thumbnail_size)
                elif event.key == pygame.K_d and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Ctrl+D to toggle sort direction
                    sort_ascending = not sort_ascending
                    save_sort_settings(sort_mode_index, sort_ascending)
                    paths = sort_image_paths(paths, sort_modes[sort_mode_index], sort_ascending)
                    # Clear cache and create new lazy loader with sorted paths
                    lazy_loader = LazyImageLoader(paths, thumbnail_size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    # Check if click is in tree panel
                    if mouse_x < tree_width:
                        # Check if click is on control buttons
                        button_y = screen_height - 70
                        if button_y <= mouse_y <= button_y + 20:
                            # Select All button
                            select_all_surface = font.render("SELECT ALL" if not select_all else "DESELECT ALL", True, TEXT_COLOR)
                            if select_all_x <= mouse_x <= select_all_x + select_all_surface.get_width():
                                select_all = not select_all
                                if select_all:
                                    selected_indices = list(range(len(paths)))
                                    selected_set = set(range(len(paths)))
                                else:
                                    selected_indices = []
                                    selected_set = set()
                                break
                            # Sort Mode button
                            elif sort_mode_x <= mouse_x <= sort_mode_x + font.render(f"Sort: {sort_modes[sort_mode_index].title()}", True, TEXT_COLOR).get_width():
                                sort_mode_index = (sort_mode_index + 1) % len(sort_modes)
                                save_sort_settings(sort_mode_index, sort_ascending)
                                paths = sort_image_paths(paths, sort_modes[sort_mode_index], sort_ascending)
                                # Clear cache and create new lazy loader with sorted paths
                                lazy_loader = LazyImageLoader(paths, thumbnail_size)
                                break
                            # Sort Direction button
                            elif sort_dir_x <= mouse_x <= sort_dir_x + font.render("↑" if sort_ascending else "↓", True, TEXT_COLOR).get_width():
                                sort_ascending = not sort_ascending
                                save_sort_settings(sort_mode_index, sort_ascending)
                                paths = sort_image_paths(paths, sort_modes[sort_mode_index], sort_ascending)
                                # Clear cache and create new lazy loader with sorted paths
                                lazy_loader = LazyImageLoader(paths, thumbnail_size)
                                break
                        else:
                            # Tree navigation click
                            tree_y = mouse_y + tree_scroll_offset
                            item_index = tree_y // 25
                            if 0 <= item_index < len(folder_tree):
                                item_name, item_path, is_folder = folder_tree[item_index]
                                if is_folder:
                                    # Navigate to folder - async loading
                                    current_folder = item_path
                                    loading_folder = True
                                    selected_indices = []
                                    selected_set = set()
                                    folder_tree = build_folder_tree(current_folder, current_folder)
                                    tree_scroll_offset = 0
                                    max_tree_scroll = max(0, len(folder_tree) * 25 - (screen_height - 100))

                                    # Start async loading
                                    def on_folder_loaded(new_paths):
                                        nonlocal paths, lazy_loader, max_scroll, loading_folder
                                        paths = new_paths
                                        lazy_loader = LazyImageLoader(paths, thumbnail_size)
                                        max_scroll = max(0, ((len(paths) + cols - 1) // cols) * cell_height - screen_height + 100)
                                        loading_folder = False

                                    loading_thread = load_folder_async(current_folder, sort_mode_index, sort_ascending, sort_modes, on_folder_loaded)
                    else:
                        # Thumbnail panel click
                        mouse_x -= tree_width
                        mouse_y += scroll_offset

                        # Check which thumbnail was clicked
                        for i, _ in enumerate(paths):
                            row = i // cols
                            col = i % cols
                            x = thumbnail_panel_x + col * cell_width + padding
                            y = row * cell_height + padding
                            if (x <= mouse_x + tree_width < x + thumb_width and
                                y <= mouse_y < y + thumb_height):
                                if i in selected_set:
                                    # Remove from selection
                                    selected_indices.remove(i)
                                    selected_set.remove(i)
                                else:
                                    # Add to selection
                                    selected_indices.append(i)
                                    selected_set.add(i)
                                break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Mouse wheel up
                        mouse_x, _ = pygame.mouse.get_pos()
                        if mouse_x < tree_width:
                            tree_scroll_offset = max(0, min(max_tree_scroll, tree_scroll_offset - 25))
                        else:
                            scroll_offset = max(0, min(max_scroll, scroll_offset - 50))
                    elif event.button == 5:  # Mouse wheel down
                        mouse_x, _ = pygame.mouse.get_pos()
                        if mouse_x < tree_width:
                            tree_scroll_offset = max(0, min(max_tree_scroll, tree_scroll_offset + 25))
                        else:
                            scroll_offset = max(0, min(max_scroll, scroll_offset + 50))

        # Clear screen
        screen.fill(BG_COLOR)

        # Draw tree panel
        pygame.draw.rect(screen, TREE_BG, (0, 0, tree_width, screen_height))
        pygame.draw.line(screen, BORDER_COLOR, (tree_width, 0), (tree_width, screen_height), 2)

        # Draw tree items
        for i, (item_name, item_path, is_folder) in enumerate(folder_tree):
            y = i * 25 - tree_scroll_offset
            if 0 <= y < screen_height - 50:
                if item_path == current_folder:
                    pygame.draw.rect(screen, TREE_SELECTED, (0, y, tree_width, 25))
                text_surface = render_text_with_outline(font, item_name, TEXT_COLOR)
                screen.blit(text_surface, (10, y + 5))

        # Draw control buttons
        button_y = screen_height - 70

        # Select All button
        select_all_text = "SELECT ALL" if not select_all else "DESELECT ALL"
        select_all_surface = font.render(select_all_text, True, TEXT_COLOR)
        select_all_x = 10
        pygame.draw.rect(screen, THUMB_BG, (select_all_x - 5, button_y - 5, select_all_surface.get_width() + 10, select_all_surface.get_height() + 10))
        screen.blit(select_all_surface, (select_all_x, button_y))

        # Sort Mode button
        sort_mode_text = f"Sort: {sort_modes[sort_mode_index].title()}"
        sort_mode_surface = font.render(sort_mode_text, True, TEXT_COLOR)
        sort_mode_x = select_all_x + select_all_surface.get_width() + 20
        pygame.draw.rect(screen, THUMB_BG, (sort_mode_x - 5, button_y - 5, sort_mode_surface.get_width() + 10, sort_mode_surface.get_height() + 10))
        screen.blit(sort_mode_surface, (sort_mode_x, button_y))

        # Sort Direction button
        sort_dir_text = "↑" if sort_ascending else "↓"
        sort_dir_surface = font.render(sort_dir_text, True, TEXT_COLOR)
        sort_dir_x = sort_mode_x + sort_mode_surface.get_width() + 20
        pygame.draw.rect(screen, THUMB_BG, (sort_dir_x - 5, button_y - 5, sort_dir_surface.get_width() + 10, sort_dir_surface.get_height() + 10))
        screen.blit(sort_dir_surface, (sort_dir_x, button_y))

        # Draw loading overlay if folder is loading
        if loading_folder:
            loading_surface = pygame.Surface((screen_width - tree_width, screen_height))
            loading_surface.set_alpha(128)  # Semi-transparent
            loading_surface.fill((0, 0, 0))
            screen.blit(loading_surface, (tree_width, 0))

            # Loading text in center
            loading_text = render_text_with_outline(font, "Loading folder contents...", (255, 255, 255))
            text_x = tree_width + (screen_width - tree_width - loading_text.get_width()) // 2
            text_y = screen_height // 2 - loading_text.get_height() // 2
            screen.blit(loading_text, (text_x, text_y))
        else:
            # Draw thumbnails (lazy loading) - optimized for smooth scrolling
            load_count = 0  # Limit thumbnail loading per frame
            max_loads_per_frame = 1  # Only load 1 new thumbnail per frame

            for i in range(len(paths)):
                row = i // cols
                col = i % cols
                x = thumbnail_panel_x + col * cell_width + padding
                y = row * cell_height + padding - scroll_offset

                # Only draw if visible on screen
                if y + cell_height < 0 or y > screen_height:
                    continue

                # Get thumbnail (lazy load if needed, but limit loading rate)
                path = paths[i]
                if path in lazy_loader.cache:
                    thumb_surface = lazy_loader.cache[path][0]
                elif load_count < max_loads_per_frame:
                    thumb_surface = lazy_loader.get_thumbnail(path)
                    if thumb_surface:  # Successfully loaded
                        load_count += 1
                    else:
                        # Still loading - show placeholder
                        thumb_surface = pygame.Surface(lazy_loader.thumbnail_size)
                        thumb_surface.fill((30, 30, 50)) # Dark blue placeholder
                        font_temp = pygame.font.SysFont('Arial', 12)
                        text_surface = render_text_with_outline(font_temp, "LOADING...", (255, 255, 255))
                        thumb_surface.blit(text_surface, (10, lazy_loader.thumbnail_size[1]//2 - 10))
                else:
                    # Loading limit reached - show placeholder
                    thumb_surface = pygame.Surface(lazy_loader.thumbnail_size)
                    thumb_surface.fill((20, 20, 40)) # Darker blue placeholder
                    font_temp = pygame.font.SysFont('Arial', 10)
                    text_surface = render_text_with_outline(font_temp, "QUEUED", (200, 200, 200))
                    thumb_surface.blit(text_surface, (5, lazy_loader.thumbnail_size[1]//2 - 8))

                # Thumbnail background
                bg_color = SELECTED_COLOR if i in selected_set else THUMB_BG
                pygame.draw.rect(screen, bg_color,
                               (x - 5, y - 5, thumb_width + 10, thumb_height + 10))
                pygame.draw.rect(screen, BORDER_COLOR,
                               (x - 5, y - 5, thumb_width + 10, thumb_height + 10), 2)

                if thumb_surface:
                    # Thumbnail loaded
                    screen.blit(thumb_surface, (x, y))
                else:
                    # Still loading - show loading indicator
                    pygame.draw.rect(screen, LOADING_COLOR, (x, y, thumb_width, thumb_height))
                    # Only render loading text every few frames to improve performance
                    if pygame.time.get_ticks() % 500 < 250:  # Blink every 500ms
                        loading_text = render_text_with_outline(font, "LOADING", TEXT_COLOR)
                        text_rect = loading_text.get_rect(center=(x + thumb_width//2, y + thumb_height//2))
                        screen.blit(loading_text, text_rect)

                # Filename - optimize by caching rendered text
                filename = os.path.basename(paths[i])
                if len(filename) > 20:
                    filename = filename[:17] + "..."
                text_surface = render_text_with_outline(font, filename, TEXT_COLOR)
                text_x = x + (thumb_width - text_surface.get_width()) // 2
                text_y = y + thumb_height + 10
                screen.blit(text_surface, (text_x, text_y))

        # Draw scroll indicator
        if max_scroll > 0:
            scroll_bar_height = screen_height * (screen_height / (max_scroll + screen_height))
            scroll_bar_y = (scroll_offset / max_scroll) * (screen_height - scroll_bar_height)
            pygame.draw.rect(screen, BORDER_COLOR, (screen_width - 10, scroll_bar_y, 8, scroll_bar_height))

        # Instructions
        selected_count = len(selected_indices)
        sort_mode_name = sort_modes[sort_mode_index].title()
        sort_dir_symbol = "↑" if sort_ascending else "↓"
        instr_text = f"Selected: {selected_count} | ESC/ENTER: finish | Ctrl+A: select all | Ctrl+S: sort mode | Ctrl+D: direction"
        instr_surface = render_text_with_outline(font, instr_text, TEXT_COLOR)
        screen.blit(instr_surface, (tree_width + 10, 10))

        # Show current sort status
        sort_status_text = f"Sort: {sort_mode_name} {sort_dir_symbol}"
        sort_status_surface = font.render(sort_status_text, True, TEXT_COLOR)
        screen.blit(sort_status_surface, (tree_width + 10, 35))

        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS for smooth scrolling

    pygame.quit()

    # Return indices of selected images in order they were clicked
    return selected_indices

def process_selected_images(images, selected_indices, target_width_param=None, target_height_param=None, padding_color="#000000"):
    """Process selected images into required formats"""
    if not selected_indices:
        # Return empty tensors if nothing selected
        empty_tensor = torch.empty(0, 0, 0, 3)
        empty_mask = torch.empty(0, 0, 0)
        return [], [], empty_tensor, empty_mask

    selected_images = [images[i] for i in selected_indices]

    # Convert to tensors
    image_list = []
    mask_list = []

    for img in selected_images:
        tensor, mask = pil_to_tensor(img)
        image_list.append(tensor)
        mask_list.append(mask)

    # Determine target dimensions
    if target_width_param and target_height_param and target_width_param > 0 and target_height_param > 0:
        # Use provided target dimensions
        target_width = target_width_param
        target_height = target_height_param
    else:
        # Find largest area image for batch sizing (original logic)
        areas = [(img.width * img.height) for img in selected_images]
        max_area_idx = areas.index(max(areas))
        target_img = selected_images[max_area_idx]

        # Calculate target size (max side 2048, maintain aspect ratio)
        target_ratio = target_img.width / target_img.height
        if target_img.width > target_img.height:
            target_width = min(2048, target_img.width)
            target_height = int(target_width / target_ratio)
        else:
            target_height = min(2048, target_img.height)
            target_width = int(target_height * target_ratio)

    # Resize all images to target size maintaining aspect ratio, pad with black
    resized_images = []
    resized_masks = []

    for tensor, mask in zip(image_list, mask_list):
        img = tensor.squeeze(0)  # (H,W,3)
        mask_2d = mask.squeeze(0)  # (H,W)

        # Resize image using PIL for better quality with (H,W,C) format
        pil_img = Image.fromarray((img.numpy() * 255).astype(np.uint8))

        # Calculate scaling to fit within target dimensions while maintaining aspect ratio
        img_ratio = pil_img.width / pil_img.height
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            # Image is wider than target aspect ratio - fit by width
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            # Image is taller than target aspect ratio - fit by height
            new_height = target_height
            new_width = int(target_height * img_ratio)

        # Resize the image maintaining aspect ratio
        resized_pil = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create canvas of target size with specified padding color
        try:
            rgb_color = hex_to_rgb(padding_color)
        except:
            rgb_color = (0, 0, 0)  # Default to black on error
        canvas = Image.new('RGB', (target_width, target_height), rgb_color)

        # Paste resized image centered on canvas
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        canvas.paste(resized_pil, (x_offset, y_offset))

        resized_img = torch.from_numpy(np.array(canvas)).float() / 255.0
        resized_images.append(resized_img.unsqueeze(0))  # (1,H,W,3)

        # Resize mask with same logic
        mask_pil = Image.fromarray((mask_2d.numpy() * 255).astype(np.uint8), mode='L')
        resized_mask_pil = mask_pil.resize((new_width, new_height), Image.Resampling.NEAREST)

        # Create black canvas for mask (0 = transparent for padding)
        mask_canvas = Image.new('L', (target_width, target_height), 0)

        # Paste resized mask centered on canvas
        mask_canvas.paste(resized_mask_pil, (x_offset, y_offset))

        resized_mask = torch.from_numpy(np.array(mask_canvas)).float() / 255.0
        resized_masks.append(resized_mask.unsqueeze(0))  # (1,H,W)

    # Batch tensors
    image_batch = torch.cat(resized_images, dim=0)  # (B, H, W, 3)
    mask_batch = torch.cat(resized_masks, dim=0)    # (B, H, W)

    return image_list, mask_list, image_batch, mask_batch

def image_selector(folder_path, target_width_param=None, target_height_param=None, padding_color="#000000"):
    """Main function: load image paths, show UI, process selection"""
    # Only load paths initially, not images
    _, paths = load_images_from_folder(folder_path)

    if not paths:
        # Return empty results if no images found
        empty_tensor = torch.empty(0, 0, 0, 3)
        empty_mask = torch.empty(0, 0, 0)
        return [], [], empty_tensor, empty_mask

    # Sort paths initially (name ascending by default)
    paths = sort_image_paths(paths, 'name', True)

    # Show selection UI (will load images lazily)
    selected_indices = create_image_selector_ui(paths, folder_path, padding_color, target_width_param, target_height_param)

    # Now load only the selected images
    selected_paths = [paths[i] for i in selected_indices]
    selected_images = []
    for path in selected_paths:
        try:
            img = Image.open(path)
            selected_images.append(img)
        except Exception as e:
            print(f"Failed to load selected image {path}: {e}")
            continue

    # Process selected images
    return process_selected_images(selected_images, list(range(len(selected_images))), target_width_param, target_height_param, padding_color)

# Module exports for ComfyUI node
__all__ = ['image_selector']
