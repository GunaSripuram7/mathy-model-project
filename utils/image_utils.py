"""
Image utility functions for mathematical art generation.
Includes image processing, visualization, and conversion utilities.
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Optional, Union
import io
import base64
import logging

logger = logging.getLogger(__name__)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a torch tensor to PIL Image"""
    if tensor.dim() == 4:
        # Batch dimension, take first image
        tensor = tensor[0]
    
    # Ensure tensor is on CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert from [-1, 1] to [0, 1] if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 3:
        # C, H, W -> H, W, C
        np_image = tensor.permute(1, 2, 0).numpy()
    else:
        # Grayscale
        np_image = tensor.numpy()
    
    # Convert to 8-bit
    np_image = (np_image * 255).astype(np.uint8)
    
    # Create PIL image
    if np_image.ndim == 3:
        image = Image.fromarray(np_image, mode='RGB')
    else:
        image = Image.fromarray(np_image, mode='L')
    
    return image

def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL Image to torch tensor"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy
    np_image = np.array(image).astype(np.float32)
    
    # Normalize to [0, 1]
    np_image = np_image / 255.0
    
    # Convert to tensor and rearrange dimensions
    tensor = torch.from_numpy(np_image).permute(2, 0, 1)  # H, W, C -> C, H, W
    
    # Normalize to [-1, 1] if requested
    if normalize:
        tensor = tensor * 2.0 - 1.0
    
    return tensor

def save_image_grid(images: List[Union[torch.Tensor, Image.Image]], 
                   filepath: str,
                   grid_size: Optional[Tuple[int, int]] = None,
                   titles: Optional[List[str]] = None,
                   title_fontsize: int = 12) -> None:
    """Save a grid of images to file"""
    
    # Convert all images to PIL
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            pil_images.append(tensor_to_pil(img))
        else:
            pil_images.append(img)
    
    num_images = len(pil_images)
    if num_images == 0:
        logger.warning("No images to save")
        return
    
    # Determine grid size
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        grid_size = (rows, cols)
    else:
        rows, cols = grid_size
    
    # Get image size (assume all images are the same size)
    img_width, img_height = pil_images[0].size
    
    # Calculate grid dimensions
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    # Add space for titles if provided
    title_height = title_fontsize + 10 if titles else 0
    grid_height += rows * title_height
    
    # Create grid image
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images into grid
    for i, img in enumerate(pil_images):
        row = i // cols
        col = i % cols
        
        x = col * img_width
        y = row * (img_height + title_height)
        
        grid_image.paste(img, (x, y))
        
        # Add title if provided
        if titles and i < len(titles):
            draw = ImageDraw.Draw(grid_image)
            title_y = y + img_height + 2
            
            try:
                # Try to use a better font
                font = ImageFont.truetype("arial.ttf", title_fontsize)
            except:
                # Fall back to default font
                font = ImageFont.load_default()
            
            draw.text((x + 5, title_y), titles[i], fill='black', font=font)
    
    # Save grid
    grid_image.save(filepath)
    logger.info(f"Saved image grid to {filepath}")

def resize_image(image: Union[torch.Tensor, Image.Image], 
                size: Tuple[int, int],
                method: str = 'bilinear') -> Union[torch.Tensor, Image.Image]:
    """Resize image to specified size"""
    
    if isinstance(image, torch.Tensor):
        # Use torch interpolation
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        
        if method == 'bilinear':
            resized = torch.nn.functional.interpolate(
                image, size=size, mode='bilinear', align_corners=False
            )
        elif method == 'nearest':
            resized = torch.nn.functional.interpolate(
                image, size=size, mode='nearest'
            )
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        return resized.squeeze(0)  # Remove batch dimension
    
    else:
        # Use PIL resize
        if method == 'bilinear':
            pil_method = Image.BILINEAR
        elif method == 'nearest':
            pil_method = Image.NEAREST
        else:
            pil_method = Image.BILINEAR
        
        return image.resize(size, pil_method)

def apply_colormap(grayscale_image: Union[torch.Tensor, np.ndarray], 
                   colormap: str = 'viridis') -> np.ndarray:
    """Apply a colormap to a grayscale image"""
    
    # Convert to numpy if needed
    if isinstance(grayscale_image, torch.Tensor):
        if grayscale_image.is_cuda:
            grayscale_image = grayscale_image.cpu()
        gray_np = grayscale_image.squeeze().numpy()
    else:
        gray_np = grayscale_image.squeeze()
    
    # Normalize to [0, 1]
    gray_np = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(gray_np)
    
    # Convert to 8-bit RGB
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return colored_rgb

def create_math_visualization(formula: str, 
                            image_size: Tuple[int, int] = (512, 512),
                            x_range: Tuple[float, float] = (-5, 5),
                            y_range: Tuple[float, float] = (-5, 5),
                            colormap: str = 'viridis',
                            dpi: int = 100) -> Image.Image:
    """Create a visualization of a mathematical formula"""
    
    try:
        # Parse and evaluate the formula
        width, height = image_size
        
        # Create coordinate grids
        x = np.linspace(x_range[0], x_range[1], width)
        y = np.linspace(y_range[0], y_range[1], height)
        X, Y = np.meshgrid(x, y)
        
        # Try to evaluate the formula
        # This is a simplified approach - a full implementation would use sympy
        result = np.zeros_like(X)
        
        # Handle some common mathematical patterns
        if 'sin' in formula and 'cos' in formula:
            # Trigonometric pattern
            result = np.sin(X) * np.cos(Y)
        elif 'x^2' in formula or 'x**2' in formula:
            # Quadratic pattern
            result = X**2 + Y**2
        elif 'spiral' in formula.lower():
            # Spiral pattern
            r = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            result = np.sin(3 * theta) * np.exp(-r/5)
        else:
            # Default pattern
            result = np.sin(np.sqrt(X**2 + Y**2))
        
        # Apply colormap
        colored = apply_colormap(result, colormap)
        
        # Create PIL image
        image = Image.fromarray(colored, mode='RGB')
        
        return image
        
    except Exception as e:
        logger.error(f"Failed to create visualization for formula '{formula}': {e}")
        
        # Return a placeholder image
        placeholder = Image.new('RGB', image_size, color='black')
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), f"Error: {formula}", fill='white')
        return placeholder

def add_formula_overlay(image: Image.Image, 
                       formula: str,
                       position: str = 'bottom',
                       background_alpha: float = 0.7,
                       text_color: str = 'white',
                       font_size: int = 16) -> Image.Image:
    """Add formula text overlay to an image"""
    
    # Create a copy of the image
    result = image.copy()
    
    # Create drawing context
    draw = ImageDraw.Draw(result)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), formula, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position
    img_width, img_height = image.size
    
    if position == 'bottom':
        x = (img_width - text_width) // 2
        y = img_height - text_height - 10
    elif position == 'top':
        x = (img_width - text_width) // 2
        y = 10
    elif position == 'center':
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
    else:
        x, y = 10, 10
    
    # Draw background rectangle if alpha > 0
    if background_alpha > 0:
        padding = 5
        bg_color = (0, 0, 0, int(255 * background_alpha))
        
        # Create overlay for transparency
        overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        overlay_draw.rectangle([
            x - padding, y - padding,
            x + text_width + padding, y + text_height + padding
        ], fill=bg_color)
        
        # Composite with original
        result = Image.alpha_composite(result.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(result)
    
    # Draw text
    draw.text((x, y), formula, fill=text_color, font=font)
    
    return result

def create_comparison_grid(original_images: List[Image.Image],
                          generated_images: List[Image.Image],
                          formulas: List[str],
                          save_path: str) -> None:
    """Create a comparison grid showing original vs generated images"""
    
    num_pairs = min(len(original_images), len(generated_images), len(formulas))
    
    if num_pairs == 0:
        logger.warning("No image pairs to compare")
        return
    
    # Create comparison pairs
    comparison_images = []
    titles = []
    
    for i in range(num_pairs):
        comparison_images.extend([original_images[i], generated_images[i]])
        titles.extend([f"Original: {formulas[i]}", f"Generated: {formulas[i]}"])
    
    # Save grid with 2 columns (original, generated)
    save_image_grid(
        comparison_images,
        save_path,
        grid_size=(num_pairs, 2),
        titles=titles
    )

def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(base64_str)
    buffer = io.BytesIO(img_data)
    image = Image.open(buffer)
    return image

def calculate_image_stats(image: Union[torch.Tensor, Image.Image]) -> dict:
    """Calculate basic statistics for an image"""
    
    if isinstance(image, Image.Image):
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
    else:
        # Convert tensor to numpy
        if image.is_cuda:
            image = image.cpu()
        img_array = image.numpy()
        
        if img_array.ndim == 3:
            # C, H, W -> H, W, C
            img_array = img_array.transpose(1, 2, 0)
    
    stats = {
        'mean': float(np.mean(img_array)),
        'std': float(np.std(img_array)),
        'min': float(np.min(img_array)),
        'max': float(np.max(img_array)),
        'shape': img_array.shape
    }
    
    # Color-specific stats for RGB images
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        stats['mean_rgb'] = [float(np.mean(img_array[:, :, i])) for i in range(3)]
        stats['std_rgb'] = [float(np.std(img_array[:, :, i])) for i in range(3)]
    
    return stats

# Example usage and testing
if __name__ == "__main__":
    # Test image utilities
    print("Testing image utilities...")
    
    # Create a test image
    test_tensor = torch.randn(3, 256, 256)
    test_pil = tensor_to_pil(test_tensor)
    
    print(f"Test image size: {test_pil.size}")
    
    # Test mathematical visualization
    test_formula = "sin(x) * cos(y)"
    math_viz = create_math_visualization(test_formula)
    
    print(f"Math visualization size: {math_viz.size}")
    
    # Test grid creation
    test_images = [test_pil, math_viz]
    test_titles = ["Random Tensor", "Math Formula"]
    
    # Would save grid here in a real scenario
    print("Image utilities test completed!")
