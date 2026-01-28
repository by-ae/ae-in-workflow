import os
import pygame
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import resize
import math

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

def create_image_selector_ui(images, paths, thumbnail_size=(200, 200)):
    """Create pygame UI for image selection"""
    pygame.init()

    # Screen setup
    screen_width = 1200
    screen_height = 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Image Selector - Click to select, ESC to finish")

    # Colors
    BG_COLOR = (30, 30, 30)
    THUMB_BG = (50, 50, 50)
    SELECTED_COLOR = (100, 150, 255)
    BORDER_COLOR = (70, 70, 70)
    TEXT_COLOR = (220, 220, 220)

    # Font
    font = pygame.font.SysFont('Arial', 16)

    # Grid setup
    cols = 4
    padding = 20
    thumb_width, thumb_height = thumbnail_size
    cell_width = thumb_width + padding * 2
    cell_height = thumb_height + padding * 2 + 40  # Extra space for filename

    # Create thumbnails
    thumbnails = []
    for img in images:
        thumb = create_thumbnail(img, thumbnail_size)
        # Ensure thumbnail is in RGB mode for pygame
        if thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')
        thumb_surface = pygame.surfarray.make_surface(
            np.transpose(np.array(thumb), (1, 0, 2))
        )
        thumbnails.append(thumb_surface)

    # Selection state - track order of selection
    selected_indices = []  # Will store indices in order they were selected
    selected_set = set()   # For quick lookup of what's selected
    scroll_offset = 0
    max_scroll = max(0, ((len(images) + cols - 1) // cols) * cell_height - screen_height + 100)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    mouse_y += scroll_offset

                    # Check which thumbnail was clicked
                    for i, _ in enumerate(images):
                        row = i // cols
                        col = i % cols
                        x = col * cell_width + padding
                        y = row * cell_height + padding
                        if (x <= mouse_x < x + thumb_width and
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
            elif event.type == pygame.MOUSEWHEEL:
                scroll_offset = max(0, min(max_scroll, scroll_offset - event.y * 50))

        # Clear screen
        screen.fill(BG_COLOR)

        # Draw thumbnails
        for i, thumb_surface in enumerate(thumbnails):
            row = i // cols
            col = i % cols
            x = col * cell_width + padding
            y = row * cell_height + padding - scroll_offset

            # Only draw if visible
            if -cell_height < y < screen_height:
                # Thumbnail background
                bg_color = SELECTED_COLOR if i in selected_set else THUMB_BG
                pygame.draw.rect(screen, bg_color,
                               (x - 5, y - 5, thumb_width + 10, thumb_height + 10))
                pygame.draw.rect(screen, BORDER_COLOR,
                               (x - 5, y - 5, thumb_width + 10, thumb_height + 10), 2)

                # Thumbnail
                screen.blit(thumb_surface, (x, y))

                # Filename
                filename = os.path.basename(paths[i])
                if len(filename) > 20:
                    filename = filename[:17] + "..."
                text_surface = font.render(filename, True, TEXT_COLOR)
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
        instr_text = f"Selected: {selected_count} | ESC or ENTER to finish"
        instr_surface = font.render(instr_text, True, TEXT_COLOR)
        screen.blit(instr_surface, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    # Return indices of selected images in order they were clicked
    return selected_indices

def process_selected_images(images, selected_indices, target_width_param=None, target_height_param=None):
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

        # Create black canvas of target size
        canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))

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

def image_selector(folder_path, target_width_param=None, target_height_param=None):
    """Main function: load images, show UI, process selection"""
    images, paths = load_images_from_folder(folder_path)

    if not images:
        # Return empty results if no images found
        empty_tensor = torch.empty(0, 0, 0, 3)
        empty_mask = torch.empty(0, 0, 0)
        return [], [], empty_tensor, empty_mask

    # Show selection UI
    selected_indices = create_image_selector_ui(images, paths)

    # Process selected images
    return process_selected_images(images, selected_indices, target_width_param, target_height_param)

# Module exports for ComfyUI node
__all__ = ['image_selector']
