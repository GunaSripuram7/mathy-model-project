"""
Formula-to-image rendering engine.
Provides fallback rendering capabilities and comparison tools for mathematical formulas.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Union, Callable
import re
import math
import logging

logger = logging.getLogger(__name__)

class MathRenderer:
    """Renders mathematical formulas as images"""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512), dpi: int = 100):
        self.image_size = image_size
        self.dpi = dpi
        self.figure_size = (image_size[0] / dpi, image_size[1] / dpi)
        
        # Default rendering parameters
        self.default_params = {
            'x_range': (-5, 5),
            'y_range': (-5, 5),
            'resolution': 1000,
            'colormap': 'viridis',
            'background_color': 'black',
            'line_width': 2.0
        }
    
    def render_parametric(self, formula_x: str, formula_y: str, 
                         t_range: Tuple[float, float] = (0, 2*np.pi),
                         num_points: int = 1000,
                         **kwargs) -> Image.Image:
        """Render parametric equations"""
        try:
            # Create parameter values
            t = np.linspace(t_range[0], t_range[1], num_points)
            
            # Evaluate equations
            x_vals = self._eval_expression(formula_x, {'t': t})
            y_vals = self._eval_expression(formula_y, {'t': t})
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot parametric curve
            colormap = kwargs.get('colormap', self.default_params['colormap'])
            if isinstance(colormap, str):
                colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(x_vals)))
            else:
                colors = colormap
            
            ax.scatter(x_vals, y_vals, c=colors, s=1, alpha=0.8)
            
            # Styling
            ax.set_facecolor(kwargs.get('background_color', self.default_params['background_color']))
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Remove axes if requested
            if kwargs.get('remove_axes', True):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
            
            # Convert to PIL Image
            image = self._matplotlib_to_pil(fig)
            plt.close(fig)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to render parametric equations: {e}")
            return self._create_error_image(f"Parametric Error: {e}")
    
    def render_polar(self, formula: str, 
                    theta_range: Tuple[float, float] = (0, 2*np.pi),
                    num_points: int = 1000,
                    **kwargs) -> Image.Image:
        """Render polar equations"""
        try:
            # Create angle values
            theta = np.linspace(theta_range[0], theta_range[1], num_points)
            
            # Evaluate radius formula
            r_vals = self._eval_expression(formula, {'theta': theta, 'pi': np.pi})
            
            # Convert to Cartesian coordinates
            x_vals = r_vals * np.cos(theta)
            y_vals = r_vals * np.sin(theta)
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot polar curve
            colormap = kwargs.get('colormap', self.default_params['colormap'])
            if isinstance(colormap, str):
                colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(x_vals)))
            else:
                colors = colormap
            
            ax.plot(x_vals, y_vals, linewidth=kwargs.get('line_width', 2.0))
            
            # Styling
            ax.set_facecolor(kwargs.get('background_color', self.default_params['background_color']))
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Remove axes if requested
            if kwargs.get('remove_axes', True):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            # Convert to PIL Image
            image = self._matplotlib_to_pil(fig)
            plt.close(fig)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to render polar equation: {e}")
            return self._create_error_image(f"Polar Error: {e}")
    
    def render_implicit(self, formula: str,
                       x_range: Optional[Tuple[float, float]] = None,
                       y_range: Optional[Tuple[float, float]] = None,
                       resolution: int = 500,
                       **kwargs) -> Image.Image:
        """Render implicit equations (like x^2 + y^2 = 1)"""
        try:
            if x_range is None:
                x_range = self.default_params['x_range']
            if y_range is None:
                y_range = self.default_params['y_range']
            
            # Create coordinate grids
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            
            # Evaluate the implicit equation
            Z = self._eval_expression(formula, {'x': X, 'y': Y})
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot contour at zero level (implicit curve)
            colormap = kwargs.get('colormap', self.default_params['colormap'])
            contour = ax.contour(X, Y, Z, levels=[0], colors='white', linewidths=2)
            
            # Add filled contour for better visualization
            if kwargs.get('fill', True):
                ax.contourf(X, Y, Z, levels=50, cmap=colormap, alpha=0.7)
            
            # Styling
            ax.set_facecolor(kwargs.get('background_color', self.default_params['background_color']))
            ax.set_aspect('equal')
            
            # Remove axes if requested
            if kwargs.get('remove_axes', True):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            # Convert to PIL Image
            image = self._matplotlib_to_pil(fig)
            plt.close(fig)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to render implicit equation: {e}")
            return self._create_error_image(f"Implicit Error: {e}")
    
    def render_3d_surface(self, formula: str,
                         x_range: Optional[Tuple[float, float]] = None,
                         y_range: Optional[Tuple[float, float]] = None,
                         resolution: int = 100,
                         **kwargs) -> Image.Image:
        """Render 3D surface as 2D projection"""
        try:
            if x_range is None:
                x_range = self.default_params['x_range']
            if y_range is None:
                y_range = self.default_params['y_range']
            
            # Create coordinate grids
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            
            # Evaluate the surface
            Z = self._eval_expression(formula, {'x': X, 'y': Y})
            
            # Create 3D plot
            fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot surface
            colormap = kwargs.get('colormap', self.default_params['colormap'])
            surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.8, 
                                 linewidth=0, antialiased=True)
            
            # Styling
            ax.set_facecolor(kwargs.get('background_color', self.default_params['background_color']))
            
            # Remove axes if requested
            if kwargs.get('remove_axes', True):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            # Set viewing angle
            ax.view_init(elev=kwargs.get('elevation', 30), azim=kwargs.get('azimuth', 45))
            
            # Convert to PIL Image
            image = self._matplotlib_to_pil(fig)
            plt.close(fig)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to render 3D surface: {e}")
            return self._create_error_image(f"3D Surface Error: {e}")
    
    def render_fractal(self, formula_type: str = "mandelbrot",
                      x_range: Tuple[float, float] = (-2, 2),
                      y_range: Tuple[float, float] = (-2, 2),
                      max_iter: int = 100,
                      resolution: int = 500,
                      **kwargs) -> Image.Image:
        """Render fractal patterns"""
        try:
            if formula_type.lower() == "mandelbrot":
                return self._render_mandelbrot(x_range, y_range, max_iter, resolution, **kwargs)
            elif formula_type.lower() == "julia":
                c = kwargs.get('c', complex(-0.4, 0.6))
                return self._render_julia(x_range, y_range, c, max_iter, resolution, **kwargs)
            else:
                return self._create_error_image(f"Unknown fractal type: {formula_type}")
                
        except Exception as e:
            logger.error(f"Failed to render fractal: {e}")
            return self._create_error_image(f"Fractal Error: {e}")
    
    def _render_mandelbrot(self, x_range, y_range, max_iter, resolution, **kwargs):
        """Render Mandelbrot set"""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        
        Z = np.zeros_like(C)
        iterations = np.zeros(C.shape, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            iterations[mask] = i
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        colormap = kwargs.get('colormap', 'hot')
        ax.imshow(iterations, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                 cmap=colormap, origin='lower')
        
        # Styling
        if kwargs.get('remove_axes', True):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        image = self._matplotlib_to_pil(fig)
        plt.close(fig)
        return image
    
    def _render_julia(self, x_range, y_range, c, max_iter, resolution, **kwargs):
        """Render Julia set"""
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        
        iterations = np.zeros(Z.shape, dtype=int)
        
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + c
            iterations[mask] = i
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        colormap = kwargs.get('colormap', 'hot')
        ax.imshow(iterations, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                 cmap=colormap, origin='lower')
        
        # Styling
        if kwargs.get('remove_axes', True):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        image = self._matplotlib_to_pil(fig)
        plt.close(fig)
        return image
    
    def _eval_expression(self, expression: str, variables: Dict[str, np.ndarray]) -> np.ndarray:
        """Safely evaluate mathematical expression"""
        
        # Create safe namespace
        safe_dict = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'sqrt': np.sqrt, 'abs': np.abs,
            'pi': np.pi, 'e': np.e,
            'power': np.power, 'pow': np.power,
            **variables
        }
        
        # Preprocess expression
        expr = expression.replace('^', '**')  # Handle power notation
        
        try:
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return np.asarray(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate expression '{expression}': {e}")
            # Return zeros as fallback
            first_var = next(iter(variables.values()))
            return np.zeros_like(first_var)
    
    def _matplotlib_to_pil(self, fig) -> Image.Image:
        """Convert matplotlib figure to PIL Image"""
        # Save figure to buffer
        from io import BytesIO
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', 
                   facecolor='black', edgecolor='none', dpi=self.dpi)
        buffer.seek(0)
        
        # Open as PIL image
        image = Image.open(buffer)
        
        # Resize to target size
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.LANCZOS)
        
        return image
    
    def _create_error_image(self, error_message: str) -> Image.Image:
        """Create an error image with the given message"""
        image = Image.new('RGB', self.image_size, color='black')
        draw = ImageDraw.Draw(image)
        
        # Try to use a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Wrap text
        words = error_message.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= self.image_size[0] - 20:  # 10px margin each side
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text
        y_offset = (self.image_size[1] - len(lines) * 20) // 2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (self.image_size[0] - (bbox[2] - bbox[0])) // 2
            y = y_offset + i * 20
            draw.text((x, y), line, fill='white', font=font)
        
        return image

class FormulaRenderer:
    """High-level interface for rendering mathematical formulas"""
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.renderer = MathRenderer(image_size)
        
        # Formula type patterns
        self.type_patterns = {
            'parametric': [r'x\s*=.*,\s*y\s*=', r'.*=.*t.*,.*=.*t'],
            'polar': [r'r\s*=.*theta', r'.*=.*sin.*theta', r'.*=.*cos.*theta'],
            'implicit': [r'.*=.*', r'x.*\+.*y.*=', r'.*x\^2.*\+.*y\^2'],
            'fractal': ['mandelbrot', 'julia', 'fractal'],
            '3d': [r'z\s*=', r'.*=.*x.*y']
        }
    
    def render_formula(self, formula: str, **kwargs) -> Image.Image:
        """Render a formula by automatically detecting its type"""
        
        formula_type = self._detect_formula_type(formula)
        
        try:
            if formula_type == 'parametric':
                return self._render_parametric_formula(formula, **kwargs)
            elif formula_type == 'polar':
                return self._render_polar_formula(formula, **kwargs)
            elif formula_type == 'implicit':
                return self._render_implicit_formula(formula, **kwargs)
            elif formula_type == 'fractal':
                return self._render_fractal_formula(formula, **kwargs)
            elif formula_type == '3d':
                return self._render_3d_formula(formula, **kwargs)
            else:
                # Default to 2D function plot
                return self._render_function_formula(formula, **kwargs)
                
        except Exception as e:
            logger.error(f"Failed to render formula '{formula}': {e}")
            return self.renderer._create_error_image(f"Render Error: {e}")
    
    def _detect_formula_type(self, formula: str) -> str:
        """Detect the type of mathematical formula"""
        formula_lower = formula.lower()
        
        for formula_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, formula_lower):
                    return formula_type
        
        return 'function'  # Default
    
    def _render_parametric_formula(self, formula: str, **kwargs) -> Image.Image:
        """Render parametric equations"""
        # Parse x = ..., y = ... format
        parts = formula.split(',')
        if len(parts) >= 2:
            x_part = parts[0].strip()
            y_part = parts[1].strip()
            
            # Extract expressions after '='
            x_expr = x_part.split('=')[1].strip() if '=' in x_part else x_part
            y_expr = y_part.split('=')[1].strip() if '=' in y_part else y_part
            
            return self.renderer.render_parametric(x_expr, y_expr, **kwargs)
        else:
            return self.renderer._create_error_image("Invalid parametric format")
    
    def _render_polar_formula(self, formula: str, **kwargs) -> Image.Image:
        """Render polar equations"""
        # Extract expression after '='
        if '=' in formula:
            expr = formula.split('=')[1].strip()
        else:
            expr = formula.strip()
        
        return self.renderer.render_polar(expr, **kwargs)
    
    def _render_implicit_formula(self, formula: str, **kwargs) -> Image.Image:
        """Render implicit equations"""
        return self.renderer.render_implicit(formula, **kwargs)
    
    def _render_fractal_formula(self, formula: str, **kwargs) -> Image.Image:
        """Render fractals"""
        if 'mandelbrot' in formula.lower():
            return self.renderer.render_fractal('mandelbrot', **kwargs)
        elif 'julia' in formula.lower():
            return self.renderer.render_fractal('julia', **kwargs)
        else:
            return self.renderer.render_fractal('mandelbrot', **kwargs)
    
    def _render_3d_formula(self, formula: str, **kwargs) -> Image.Image:
        """Render 3D surfaces"""
        # Extract expression after '='
        if '=' in formula:
            expr = formula.split('=')[1].strip()
        else:
            expr = formula.strip()
        
        return self.renderer.render_3d_surface(expr, **kwargs)
    
    def _render_function_formula(self, formula: str, **kwargs) -> Image.Image:
        """Render 2D function plots"""
        # This would implement 2D function plotting
        # For now, return a simple pattern
        return self.renderer.render_implicit(f"y - ({formula})", **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Test the renderer
    renderer = FormulaRenderer(image_size=(400, 400))
    
    test_formulas = [
        "r = sin(3*theta)",  # Polar
        "x = t*cos(t), y = t*sin(t)",  # Parametric
        "x^2 + y^2 = 1",  # Implicit
        "z = x^2 + y^2",  # 3D
        "mandelbrot"  # Fractal
    ]
    
    print("Testing Formula Renderer:")
    for i, formula in enumerate(test_formulas):
        try:
            image = renderer.render_formula(formula)
            print(f"✓ Formula {i+1}: {formula} - Rendered successfully")
            # In a real scenario, would save: image.save(f"test_{i+1}.png")
        except Exception as e:
            print(f"✗ Formula {i+1}: {formula} - Failed: {e}")
    
    print("Renderer tests completed!")
