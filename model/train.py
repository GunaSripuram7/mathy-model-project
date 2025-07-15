"""
Training script for mathematical art generation model.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import argparse
from tqdm import tqdm
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import local modules
try:
    from model.architecture import create_model
    from model.loss_functions import CombinedLoss
    from model.dataloader import create_validation_split
    from utils.image_utils import save_image_grid, tensor_to_pil
    from utils.math_parser import FormulaTokenizer
except ImportError as e:
    print(f"Warning: Could not import local modules: {e}")
    print("Some functionality may not be available.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MathArtTrainer:
    """Main trainer class for mathematical art generation"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.global_step = 0
        self.current_epoch = 0
        
        # Initialize components
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = self._create_loss_function()
        
        # Data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Logging and monitoring
        self.logger = self._setup_logging()
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config['checkpointing']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Model parameters: {self._count_parameters():,}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        device_config = self.config['hardware']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        return device
    
    def _create_model(self) -> nn.Module:
        """Create and initialize the model"""
        model_config = self.config['model']
        model = create_model(model_config)
        model.to(self.device)
        
        # Enable mixed precision if configured
        if self.config['hardware']['mixed_precision']:
            try:
                from torch.cuda.amp import autocast
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("Mixed precision training enabled")
            except ImportError:
                logger.warning("Mixed precision not available, falling back to FP32")
                self.use_amp = False
        else:
            self.use_amp = False
        
        # Model compilation (PyTorch 2.0+)
        if self.config['hardware']['compile_model']:
            try:
                model = torch.compile(model)
                logger.info("Model compilation enabled")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        training_config = self.config['training']
        
        if training_config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                betas=(training_config['adam_beta1'], training_config['adam_beta2']),
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                betas=(training_config['adam_beta1'], training_config['adam_beta2']),
                weight_decay=training_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        training_config = self.config['training']
        
        if training_config['lr_scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=training_config['learning_rate'] * training_config['min_lr_ratio']
            )
        elif training_config['lr_scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config.get('step_size', 30),
                gamma=training_config.get('gamma', 0.1)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function"""
        loss_config = self.config['loss']
        return CombinedLoss(**loss_config)
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        data_config = self.config['data']
        training_config = self.config['training']
        
        try:
            # Create tokenizer if needed
            tokenizer = None  # Would initialize proper tokenizer here
            
            train_loader, val_loader = create_validation_split(
                dataset_dir=data_config['dataset_dir'],
                validation_ratio=self.config['validation']['validation_split']
            )
            
            logger.info(f"Training batches: {len(train_loader)}")
            logger.info(f"Validation batches: {len(val_loader)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            # Create dummy loaders for testing
            from torch.utils.data import TensorDataset
            
            dummy_images = torch.randn(16, 3, 512, 512)
            dummy_tokens = torch.randint(0, 1000, (16, 256))
            dummy_dataset = TensorDataset(dummy_images, dummy_tokens)
            
            dummy_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
            return dummy_loader, dummy_loader
    
    def _setup_logging(self):
        """Setup experiment logging"""
        logging_config = self.config['logging']
        
        # Create log directory
        log_dir = Path(logging_config['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers (would setup wandb, tensorboard here)
        logger.info("Logging setup complete")
        
        return None
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        # Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast():
                model_output = self.model(
                    batch['noisy_images'],
                    batch['timesteps'],
                    batch['formula_tokens'],
                    batch.get('attention_mask')
                )
                
                # Prepare targets
                targets = {
                    'target_noise': batch['noise'],
                    'timesteps': batch['timesteps'],
                    'formula_tokens': batch['formula_tokens'],
                    'target_image': batch['image']
                }
                
                loss, loss_dict = self.loss_fn(model_output, targets)
        else:
            model_output = self.model(
                batch['noisy_images'],
                batch['timesteps'],
                batch['formula_tokens'],
                batch.get('attention_mask')
            )
            
            targets = {
                'target_noise': batch['noise'],
                'timesteps': batch['timesteps'],
                'formula_tokens': batch['formula_tokens'],
                'target_image': batch['image']
            }
            
            loss, loss_dict = self.loss_fn(model_output, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip_norm'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizer.step()
        
        # Convert losses to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in loss_dict.items()}
        
        return loss_dict
    
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                
                # Forward pass
                model_output = self.model(
                    batch['noisy_images'],
                    batch['timesteps'],
                    batch['formula_tokens'],
                    batch.get('attention_mask')
                )
                
                targets = {
                    'target_noise': batch['noise'],
                    'timesteps': batch['timesteps'],
                    'formula_tokens': batch['formula_tokens'],
                    'target_image': batch['image']
                }
                
                loss, loss_dict = self.loss_fn(model_output, targets)
                val_losses.append(loss_dict)
        
        # Average validation losses
        avg_losses = {}
        if val_losses:
            for key in val_losses[0].keys():
                avg_losses[f"val_{key}"] = np.mean([loss[key] for loss in val_losses])
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Training epoch
            self.model.train()
            epoch_losses = []
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                step_losses = self.train_step(batch)
                epoch_losses.append(step_losses)
                
                self.global_step += 1
                
                # Update progress bar
                if step_losses:
                    progress_bar.set_postfix({
                        'loss': f"{step_losses.get('total', 0):.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })
                
                # Log training step
                if self.global_step % self.config['logging']['log_every_n_steps'] == 0:
                    self._log_metrics(step_losses, step=self.global_step, prefix="train")
            
            # Average epoch losses
            avg_epoch_losses = {}
            if epoch_losses:
                for key in epoch_losses[0].keys():
                    avg_epoch_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            # Validation
            if (epoch + 1) % self.config['validation']['validate_every_n_epochs'] == 0:
                val_losses = self.validate()
                
                # Check if best model
                current_val_loss = val_losses.get('val_total', float('inf'))
                is_best = current_val_loss < best_val_loss
                if is_best:
                    best_val_loss = current_val_loss
                
                # Log validation metrics
                self._log_metrics(val_losses, step=self.global_step, prefix="val")
                
                logger.info(f"Epoch {epoch+1} - Train Loss: {avg_epoch_losses.get('total', 0):.4f}, "
                          f"Val Loss: {current_val_loss:.4f}")
            else:
                is_best = False
                logger.info(f"Epoch {epoch+1} - Train Loss: {avg_epoch_losses.get('total', 0):.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpointing']['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch + 1, is_best)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
        
        logger.info("Training completed!")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics to console and external loggers"""
        # Console logging
        if metrics:
            logger.info(f"Step {step} - {prefix} metrics: {metrics}")
        
        # Would implement wandb/tensorboard logging here
        pass

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Mathematical Art Generation Model")
    parser.add_argument("--config", type=str, default="model/config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MathArtTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        # Would implement checkpoint loading here
        logger.info(f"Resuming from checkpoint: {args.resume}")
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
