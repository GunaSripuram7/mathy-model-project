"""
Custom loss functions for mathematical art generation.
Includes perceptual loss, mathematical-aware loss, and other specialized losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, Tuple
import math

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features"""
    
    def __init__(self, feature_layers: list = [3, 8, 15, 22], weights: list = [1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Create feature extractors
        self.features = nn.ModuleList()
        start_idx = 0
        
        for layer_idx in feature_layers:
            self.features.append(vgg[start_idx:layer_idx+1])
            start_idx = layer_idx + 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Target images [B, C, H, W]
        """
        # Normalize to [0, 1] if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = (pred + 1) / 2  # Assume [-1, 1] range
        if target.min() < 0 or target.max() > 1:
            target = (target + 1) / 2
        
        # VGG expects 3-channel input
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize for VGG (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        loss = 0.0
        pred_features = pred
        target_features = target
        
        for i, (feature_extractor, weight) in enumerate(zip(self.features, self.weights)):
            pred_features = feature_extractor(pred_features)
            target_features = feature_extractor(target_features)
            
            loss += weight * F.mse_loss(pred_features, target_features)
        
        return loss

class MathematicalAwareLoss(nn.Module):
    """Loss that incorporates mathematical understanding of formulas"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha  # Weight for formula reconstruction
        self.beta = beta    # Weight for mathematical consistency
        
    def formula_reconstruction_loss(self, predicted_formula: torch.Tensor, 
                                   target_formula: torch.Tensor, 
                                   formula_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-entropy loss for formula reconstruction"""
        loss = F.cross_entropy(
            predicted_formula.view(-1, predicted_formula.size(-1)),
            target_formula.view(-1),
            reduction='none'
        )
        
        if formula_mask is not None:
            loss = loss * formula_mask.view(-1)
            return loss.sum() / formula_mask.sum()
        
        return loss.mean()
    
    def mathematical_consistency_loss(self, formula_encoding: Dict, 
                                    predicted_math_type: torch.Tensor,
                                    target_math_type: torch.Tensor) -> torch.Tensor:
        """Loss for mathematical type classification"""
        return F.cross_entropy(predicted_math_type, target_math_type)
    
    def complexity_loss(self, predicted_complexity: torch.Tensor,
                       target_complexity: torch.Tensor) -> torch.Tensor:
        """MSE loss for complexity prediction"""
        return F.mse_loss(predicted_complexity, target_complexity)
    
    def forward(self, model_output: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate mathematical-aware loss.
        
        Args:
            model_output: Dictionary containing model predictions
            targets: Dictionary containing target values
        """
        losses = {}
        total_loss = 0.0
        
        # Formula reconstruction loss
        if 'formula_reconstruction' in model_output and 'formula_tokens' in targets:
            formula_loss = self.formula_reconstruction_loss(
                model_output['formula_reconstruction'],
                targets['formula_tokens'],
                targets.get('formula_mask')
            )
            losses['formula_reconstruction'] = formula_loss
            total_loss += self.alpha * formula_loss
        
        # Mathematical type classification loss
        if 'math_type_logits' in model_output.get('formula_encoding', {}) and 'math_type' in targets:
            math_type_loss = self.mathematical_consistency_loss(
                model_output['formula_encoding'],
                model_output['formula_encoding']['math_type_logits'],
                targets['math_type']
            )
            losses['math_type'] = math_type_loss
            total_loss += self.beta * math_type_loss
        
        # Complexity prediction loss
        if 'complexity_score' in model_output.get('formula_encoding', {}) and 'complexity' in targets:
            complexity_loss = self.complexity_loss(
                model_output['formula_encoding']['complexity_score'].squeeze(),
                targets['complexity']
            )
            losses['complexity'] = complexity_loss
            total_loss += self.beta * complexity_loss
        
        return total_loss, losses

class DiffusionLoss(nn.Module):
    """Loss function for diffusion model training"""
    
    def __init__(self, loss_type: str = 'mse', huber_delta: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.huber_delta = huber_delta
    
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor, 
                timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate diffusion loss between predicted and target noise.
        
        Args:
            predicted_noise: Model's noise prediction [B, C, H, W]
            target_noise: Ground truth noise [B, C, H, W]
            timesteps: Diffusion timesteps [B] (optional, for weighting)
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == 'mae':
            loss = F.l1_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(predicted_noise, target_noise, 
                               delta=self.huber_delta, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Average over spatial dimensions
        loss = loss.mean(dim=[1, 2, 3])
        
        # Optional timestep weighting (focus on harder timesteps)
        if timesteps is not None:
            # Weight higher for intermediate timesteps
            weights = 1.0 / (1.0 + torch.exp(-0.01 * (timesteps - 500)))
            loss = loss * weights
        
        return loss.mean()

class GradientPenalty(nn.Module):
    """Gradient penalty for adversarial training (if using GAN components)"""
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, real_samples: torch.Tensor, fake_samples: torch.Tensor, 
                discriminator: nn.Module) -> torch.Tensor:
        """Calculate gradient penalty"""
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated = discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * penalty

class ColorHarmonyLoss(nn.Module):
    """Loss to encourage pleasing color combinations in mathematical art"""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
    def rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV color space"""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        max_val, max_idx = torch.max(rgb, dim=1)
        min_val, _ = torch.min(rgb, dim=1)
        
        delta = max_val - min_val
        
        # Hue calculation
        hue = torch.zeros_like(max_val)
        mask = delta != 0
        
        # Red is max
        red_mask = (max_idx == 0) & mask
        hue[red_mask] = ((g[red_mask] - b[red_mask]) / delta[red_mask]) % 6
        
        # Green is max
        green_mask = (max_idx == 1) & mask
        hue[green_mask] = (b[green_mask] - r[green_mask]) / delta[green_mask] + 2
        
        # Blue is max
        blue_mask = (max_idx == 2) & mask
        hue[blue_mask] = (r[blue_mask] - g[blue_mask]) / delta[blue_mask] + 4
        
        hue = hue * 60  # Convert to degrees
        
        # Saturation
        saturation = torch.zeros_like(max_val)
        saturation[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]
        
        # Value
        value = max_val
        
        return torch.stack([hue, saturation, value], dim=1)
    
    def color_harmony_score(self, hsv: torch.Tensor) -> torch.Tensor:
        """Calculate color harmony score based on hue relationships"""
        hue = hsv[:, 0]  # [B, H, W]
        
        # Calculate hue variance (lower is more harmonious)
        hue_mean = hue.mean(dim=[1, 2], keepdim=True)
        hue_var = ((hue - hue_mean) ** 2).mean(dim=[1, 2])
        
        # Penalize high variance (prefer harmonious colors)
        harmony_loss = hue_var.mean()
        
        return harmony_loss
    
    def forward(self, predicted_image: torch.Tensor) -> torch.Tensor:
        """Calculate color harmony loss"""
        # Ensure values are in [0, 1]
        if predicted_image.min() < 0 or predicted_image.max() > 1:
            predicted_image = torch.clamp((predicted_image + 1) / 2, 0, 1)
        
        # Convert to HSV and calculate harmony
        hsv = self.rgb_to_hsv(predicted_image)
        harmony_loss = self.color_harmony_score(hsv)
        
        return self.weight * harmony_loss

class CombinedLoss(nn.Module):
    """Combined loss function for training the mathematical art model"""
    
    def __init__(self, 
                 diffusion_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 mathematical_weight: float = 0.5,
                 color_harmony_weight: float = 0.05,
                 use_perceptual: bool = True,
                 use_mathematical: bool = True,
                 use_color_harmony: bool = True):
        super().__init__()
        
        self.diffusion_weight = diffusion_weight
        self.perceptual_weight = perceptual_weight
        self.mathematical_weight = mathematical_weight
        self.color_harmony_weight = color_harmony_weight
        
        # Initialize loss components
        self.diffusion_loss = DiffusionLoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
            
        if use_mathematical:
            self.mathematical_loss = MathematicalAwareLoss()
        else:
            self.mathematical_loss = None
            
        if use_color_harmony:
            self.color_harmony_loss = ColorHarmonyLoss()
        else:
            self.color_harmony_loss = None
    
    def forward(self, model_output: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate combined loss.
        
        Args:
            model_output: Model predictions
            targets: Target values
        """
        losses = {}
        total_loss = 0.0
        
        # Main diffusion loss
        if 'predicted_noise' in model_output and 'target_noise' in targets:
            diffusion_loss = self.diffusion_loss(
                model_output['predicted_noise'],
                targets['target_noise'],
                targets.get('timesteps')
            )
            losses['diffusion'] = diffusion_loss
            total_loss += self.diffusion_weight * diffusion_loss
        
        # Perceptual loss (when we have clean images)
        if self.perceptual_loss and 'clean_image' in model_output and 'target_image' in targets:
            perceptual_loss = self.perceptual_loss(
                model_output['clean_image'],
                targets['target_image']
            )
            losses['perceptual'] = perceptual_loss
            total_loss += self.perceptual_weight * perceptual_loss
        
        # Mathematical-aware loss
        if self.mathematical_loss:
            math_loss, math_loss_dict = self.mathematical_loss(model_output, targets)
            losses.update({f'math_{k}': v for k, v in math_loss_dict.items()})
            total_loss += self.mathematical_weight * math_loss
        
        # Color harmony loss
        if self.color_harmony_loss and 'predicted_image' in model_output:
            harmony_loss = self.color_harmony_loss(model_output['predicted_image'])
            losses['color_harmony'] = harmony_loss
            total_loss += self.color_harmony_weight * harmony_loss
        
        losses['total'] = total_loss
        return total_loss, losses
