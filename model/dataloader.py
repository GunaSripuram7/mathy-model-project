"""
Dataset loader for mathematical art generation.
Handles loading images and their associated mathematical formulas.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class MathArtDataset(Dataset):
    """Dataset for mathematical art generation"""
    
    def __init__(self, 
                 dataset_dir: str,
                 image_size: int = 512,
                 tokenizer = None,
                 max_formula_length: int = 256,
                 augmentation: bool = True,
                 normalize: bool = True):
        
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_formula_length = max_formula_length
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Setup image transforms
        self.transforms = self._setup_transforms(augmentation, normalize)
        
        logger.info(f"Loaded {len(self.metadata)} entries from {dataset_dir}")
    
    def _load_metadata(self) -> List[Dict]:
        """Load metadata from JSONL file"""
        metadata_file = self.dataset_dir / "metadata.jsonl"
        metadata = []
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        # Validate entry has required fields
                        if all(key in entry for key in ['image_path', 'formula', 'description']):
                            # Check if image exists
                            image_path = self.dataset_dir / entry['image_path']
                            if image_path.exists():
                                metadata.append(entry)
                            else:
                                logger.warning(f"Image not found: {image_path}")
                        else:
                            logger.warning(f"Invalid entry (missing required fields): {entry}")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse metadata line: {e}")
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
        
        return metadata
    
    def _setup_transforms(self, augmentation: bool, normalize: bool) -> transforms.Compose:
        """Setup image preprocessing transforms"""
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Data augmentation
        if augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                )
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        if normalize:
            # Normalize to [-1, 1] range for diffusion models
            transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        
        return transforms.Compose(transform_list)
    
    def _tokenize_formula(self, formula: str) -> Dict[str, torch.Tensor]:
        """Tokenize mathematical formula"""
        if self.tokenizer is None:
            # Simple character-level tokenization as fallback
            chars = list(formula)[:self.max_formula_length]
            
            # Create a simple vocabulary (this would be replaced with proper tokenizer)
            vocab = {char: i for i, char in enumerate(set(chars))}
            vocab['[PAD]'] = len(vocab)
            vocab['[UNK]'] = len(vocab)
            
            tokens = [vocab.get(char, vocab['[UNK]']) for char in chars]
            
            # Pad to max length
            if len(tokens) < self.max_formula_length:
                tokens.extend([vocab['[PAD]']] * (self.max_formula_length - len(tokens)))
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            attention_mask = (tokens != vocab['[PAD]']).long()
            
            return {
                'input_ids': tokens,
                'attention_mask': attention_mask
            }
        else:
            # Use provided tokenizer
            return self.tokenizer(
                formula,
                max_length=self.max_formula_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
    
    def _extract_mathematical_features(self, entry: Dict) -> Dict:
        """Extract mathematical features from metadata"""
        features = {}
        
        # Formula type classification
        formula = entry.get('formula', '').lower()
        tags = entry.get('tags', [])
        
        # Simple type classification (could be made more sophisticated)
        math_types = ['trigonometric', 'polynomial', 'exponential', 'logarithmic', 
                     'parametric', 'polar', 'algebraic', 'fractal', 'spiral', 'geometric']
        
        type_vector = torch.zeros(len(math_types))
        for i, math_type in enumerate(math_types):
            if any(math_type in tag.lower() for tag in tags) or math_type in formula:
                type_vector[i] = 1.0
        
        features['math_type'] = type_vector
        
        # Complexity estimation (simple heuristic)
        complexity = len(formula) / 100.0  # Normalize by typical formula length
        complexity += len([char for char in formula if char in '()^*/+-']) / 10.0
        complexity = min(complexity, 1.0)  # Cap at 1.0
        
        features['complexity'] = torch.tensor(complexity, dtype=torch.float32)
        
        # Color scheme encoding
        color_scheme = entry.get('color_scheme', 'viridis')
        color_schemes = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'spring', 'summer', 'autumn', 'winter']
        color_idx = color_schemes.index(color_scheme) if color_scheme in color_schemes else 0
        features['color_scheme'] = torch.tensor(color_idx, dtype=torch.long)
        
        return features
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        entry = self.metadata[idx]
        
        # Load and process image
        image_path = self.dataset_dir / entry['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transforms(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Tokenize formula
        formula_tokens = self._tokenize_formula(entry['formula'])
        
        # Extract mathematical features
        math_features = self._extract_mathematical_features(entry)
        
        # Prepare return dictionary
        item = {
            'image': image,
            'formula_tokens': formula_tokens['input_ids'].squeeze(0),
            'attention_mask': formula_tokens['attention_mask'].squeeze(0),
            'formula_text': entry['formula'],
            'description': entry['description'],
            'tags': entry.get('tags', []),
            **math_features
        }
        
        # Add any additional metadata
        if 'parameters' in entry:
            item['parameters'] = entry['parameters']
        
        return item

class DiffusionDataCollator:
    """Custom data collator for diffusion training"""
    
    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch and prepare for diffusion training"""
        # Standard collation
        collated = {}
        for key in batch[0].keys():
            if key in ['tags', 'parameters', 'description', 'formula_text']:
                # Keep as list for these fields
                collated[key] = [item[key] for item in batch]
            else:
                # Stack tensors
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Add diffusion-specific data
        batch_size = len(batch)
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,))
        
        # Sample noise
        noise = torch.randn_like(collated['image'])
        
        # Create noisy images (simplified DDPM forward process)
        # This would normally use a proper noise scheduler
        alpha_t = 1.0 - (timesteps.float() / self.num_timesteps)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        noisy_images = alpha_t * collated['image'] + (1 - alpha_t) * noise
        
        collated.update({
            'timesteps': timesteps,
            'noise': noise,
            'noisy_images': noisy_images
        })
        
        return collated

def create_dataloader(dataset_dir: str,
                     batch_size: int = 8,
                     num_workers: int = 4,
                     shuffle: bool = True,
                     image_size: int = 512,
                     tokenizer = None,
                     augmentation: bool = True,
                     **kwargs) -> DataLoader:
    """Create a DataLoader for the mathematical art dataset"""
    
    dataset = MathArtDataset(
        dataset_dir=dataset_dir,
        image_size=image_size,
        tokenizer=tokenizer,
        augmentation=augmentation,
        **kwargs
    )
    
    collator = DiffusionDataCollator()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # For stable batch sizes in training
    )
    
    return dataloader

def create_validation_split(dataset_dir: str, 
                           validation_ratio: float = 0.1,
                           random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation split"""
    
    # Load full dataset
    full_dataset = MathArtDataset(dataset_dir=dataset_dir, augmentation=False)
    
    # Create split
    total_size = len(full_dataset)
    val_size = int(total_size * validation_ratio)
    train_size = total_size - val_size
    
    torch.manual_seed(random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create separate datasets with appropriate augmentation
    train_dataset.dataset.transforms = full_dataset._setup_transforms(
        augmentation=True, normalize=True
    )
    val_dataset.dataset.transforms = full_dataset._setup_transforms(
        augmentation=False, normalize=True
    )
    
    # Create data loaders
    collator = DiffusionDataCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Test the dataset loader
    dataset_dir = "dataset"
    
    if os.path.exists(dataset_dir):
        # Create dataset
        dataset = MathArtDataset(dataset_dir)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test getting an item
            item = dataset[0]
            print(f"Sample item keys: {item.keys()}")
            print(f"Image shape: {item['image'].shape}")
            print(f"Formula: {item['formula_text']}")
            print(f"Description: {item['description']}")
        
        # Test dataloader
        dataloader = create_dataloader(dataset_dir, batch_size=2)
        
        for batch in dataloader:
            print(f"Batch keys: {batch.keys()}")
            print(f"Batch image shape: {batch['image'].shape}")
            print(f"Batch timesteps: {batch['timesteps']}")
            break
    else:
        print(f"Dataset directory not found: {dataset_dir}")
        print("Please run the ingestion scripts first.")
