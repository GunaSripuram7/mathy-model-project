#!/usr/bin/env python3
"""
Prompt-to-image generator script for mathematical art.
Generates images from mathematical formulas and text prompts using the trained model.
"""

import os
import sys
import torch
import argparse
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Union
import logging
from PIL import Image
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from model.architecture import create_model
    from utils.tokenizer import MathTokenizer
    from utils.image_utils import tensor_to_pil, save_image_grid
    from utils.render_engine import FormulaRenderer
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Some functionality may not be available.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MathArtGenerator:
    """Generator for creating mathematical art from prompts"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None):
        
        self.device = self._setup_device(device)
        self.config = self._load_config(config_path) if config_path else {}
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Fallback renderer for comparison
        self.fallback_renderer = FormulaRenderer()
        
        # Diffusion parameters
        self.num_inference_steps = self.config.get('inference', {}).get('num_inference_steps', 50)
        self.guidance_scale = self.config.get('inference', {}).get('guidance_scale', 7.5)
        
        logger.info("Mathematical Art Generator initialized successfully")
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device"""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model from config
            model_config = checkpoint.get('config', {}).get('model', {})
            if not model_config:
                # Use default config
                model_config = {
                    'vocab_size': 10000,
                    'formula_dim': 512,
                    'image_size': 512
                }
            
            model = create_model(model_config)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            # Return a dummy model for testing
            logger.warning("Using dummy model for testing")
            return self._create_dummy_model()
    
    def _create_dummy_model(self) -> torch.nn.Module:
        """Create a dummy model for testing when real model isn't available"""
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(1, 1)
            
            def forward(self, *args, **kwargs):
                # Return random noise as prediction
                batch_size = args[0].shape[0] if args else 1
                return {
                    'predicted_noise': torch.randn(batch_size, 3, 512, 512, device=self.device)
                }
        
        return DummyModel().to(self.device)
    
    def _load_tokenizer(self) -> MathTokenizer:
        """Load or create tokenizer"""
        try:
            tokenizer = MathTokenizer()
            # Would load trained tokenizer here
            logger.info("Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            return MathTokenizer()  # Use default tokenizer
    
    def generate_from_formula(self, 
                             formula: str,
                             num_samples: int = 1,
                             seed: Optional[int] = None,
                             save_path: Optional[str] = None) -> List[Image.Image]:
        """Generate images from a mathematical formula"""
        
        if seed is not None:
            torch.manual_seed(seed)
        
        logger.info(f"Generating {num_samples} images from formula: {formula}")
        
        try:
            # Tokenize formula
            encoded = self.tokenizer.encode(formula)
            formula_tokens = torch.tensor(encoded['input_ids']).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0).to(self.device)
            
            # Repeat for batch
            if num_samples > 1:
                formula_tokens = formula_tokens.repeat(num_samples, 1)
                attention_mask = attention_mask.repeat(num_samples, 1)
            
            # Generate images using diffusion process
            images = self._diffusion_sampling(formula_tokens, attention_mask, num_samples)
            
            # Convert to PIL images
            pil_images = [tensor_to_pil(img) for img in images]
            
            # Save if path provided
            if save_path:
                if num_samples == 1:
                    pil_images[0].save(save_path)
                else:
                    save_image_grid(pil_images, save_path, 
                                   titles=[f"Sample {i+1}: {formula}" for i in range(num_samples)])
                logger.info(f"Images saved to {save_path}")
            
            return pil_images
            
        except Exception as e:
            logger.error(f"Failed to generate from formula '{formula}': {e}")
            # Return fallback rendered image
            fallback = self.fallback_renderer.render_formula(formula)
            return [fallback] * num_samples
    
    def generate_from_text(self,
                          text_prompt: str,
                          num_samples: int = 1,
                          seed: Optional[int] = None,
                          save_path: Optional[str] = None) -> List[Image.Image]:
        """Generate images from descriptive text prompt"""
        
        # Try to extract mathematical formula from text or convert text to formula
        formula = self._text_to_formula(text_prompt)
        
        logger.info(f"Converting text prompt '{text_prompt}' to formula: {formula}")
        
        return self.generate_from_formula(formula, num_samples, seed, save_path)
    
    def _text_to_formula(self, text: str) -> str:
        """Convert descriptive text to mathematical formula"""
        text_lower = text.lower()
        
        # Simple text-to-formula mapping
        if 'spiral' in text_lower:
            if 'golden' in text_lower or 'fibonacci' in text_lower:
                return "r = theta^1.618"
            else:
                return "r = theta"
        
        elif 'rose' in text_lower or 'petal' in text_lower:
            # Extract number of petals if mentioned
            import re
            number_match = re.search(r'(\d+)', text)
            n = int(number_match.group(1)) if number_match else 3
            return f"r = sin({n}*theta)"
        
        elif 'circle' in text_lower:
            return "x^2 + y^2 = 1"
        
        elif 'wave' in text_lower or 'sine' in text_lower:
            return "y = sin(x)"
        
        elif 'heart' in text_lower:
            return "x = 16*sin(t)^3, y = 13*cos(t) - 5*cos(2*t) - 2*cos(3*t) - cos(4*t)"
        
        elif 'mandelbrot' in text_lower or 'fractal' in text_lower:
            return "mandelbrot"
        
        elif 'parabola' in text_lower:
            return "y = x^2"
        
        else:
            # Default to the text itself (might be a formula)
            return text
    
    def _diffusion_sampling(self, 
                           formula_tokens: torch.Tensor,
                           attention_mask: torch.Tensor,
                           num_samples: int) -> List[torch.Tensor]:
        """Perform diffusion sampling to generate images"""
        
        batch_size = formula_tokens.shape[0]
        image_size = 512  # Should match model config
        
        # Start with random noise
        images = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        
        # Simplified diffusion process (would use proper scheduler in real implementation)
        for step in range(self.num_inference_steps):
            # Create timestep
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                # Predict noise
                model_output = self.model(images, t, formula_tokens, attention_mask)
                predicted_noise = model_output.get('predicted_noise', torch.zeros_like(images))
                
                # Simple denoising step (would use proper DDPM/DDIM in real implementation)
                alpha = 1.0 - (step / self.num_inference_steps)
                images = alpha * images + (1 - alpha) * predicted_noise
        
        # Clamp to valid range
        images = torch.clamp(images, -1, 1)
        
        return [images[i] for i in range(batch_size)]
    
    def batch_generate(self,
                      prompts: List[str],
                      output_dir: str,
                      num_samples_per_prompt: int = 1,
                      seed: Optional[int] = None) -> None:
        """Generate images for multiple prompts"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        total_prompts = len(prompts)
        logger.info(f"Generating images for {total_prompts} prompts")
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{total_prompts}: {prompt}")
            
            try:
                # Generate images
                images = self.generate_from_formula(prompt, num_samples_per_prompt, seed)
                
                # Save images
                if num_samples_per_prompt == 1:
                    save_path = output_path / f"prompt_{i+1:03d}.png"
                    images[0].save(save_path)
                else:
                    save_path = output_path / f"prompt_{i+1:03d}_grid.png"
                    save_image_grid(images, str(save_path),
                                   titles=[f"Sample {j+1}" for j in range(num_samples_per_prompt)])
                
                logger.info(f"Saved to {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to process prompt '{prompt}': {e}")
    
    def create_comparison(self,
                         formula: str,
                         output_path: str,
                         num_generated: int = 3) -> None:
        """Create comparison between generated and fallback rendered images"""
        
        # Generate with model
        generated_images = self.generate_from_formula(formula, num_generated)
        
        # Create fallback rendering
        fallback_image = self.fallback_renderer.render_formula(formula)
        
        # Combine for comparison
        all_images = [fallback_image] + generated_images
        titles = ["Fallback Render"] + [f"Generated {i+1}" for i in range(num_generated)]
        
        save_image_grid(all_images, output_path, titles=titles)
        logger.info(f"Comparison saved to {output_path}")

def main():
    """Main generation script"""
    parser = argparse.ArgumentParser(description="Generate Mathematical Art from Prompts")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to model configuration file")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Mathematical formula or text prompt")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="File containing multiple prompts (one per line)")
    parser.add_argument("--output", type=str, default="generated_output.png",
                       help="Output file path")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, cpu, mps)")
    parser.add_argument("--comparison", action="store_true",
                       help="Create comparison with fallback renderer")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.prompt and not args.prompts_file:
        parser.error("Either --prompt or --prompts_file must be specified")
    
    # Create generator
    try:
        generator = MathArtGenerator(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return
    
    # Generate images
    try:
        if args.prompt:
            # Single prompt
            if args.comparison:
                generator.create_comparison(args.prompt, args.output, args.num_samples)
            else:
                images = generator.generate_from_formula(
                    args.prompt, 
                    args.num_samples, 
                    args.seed, 
                    args.output
                )
                logger.info(f"Generated {len(images)} images")
        
        elif args.prompts_file:
            # Multiple prompts from file
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            output_dir = Path(args.output).parent / "batch_output"
            generator.batch_generate(prompts, str(output_dir), args.num_samples, args.seed)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return
    
    logger.info("Generation completed successfully!")

if __name__ == "__main__":
    main()
