#!/usr/bin/env python3
"""
Script to download and curate mathematical art from various sources.
This script handles the collection and processing of mathematical art images
along with their associated formulas and metadata.
"""

import os
import json
import requests
import hashlib
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MathArtIngestor:
    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.metadata_file = self.dataset_dir / "metadata.jsonl"
        
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def download_image(self, url, filename=None):
        """Download an image from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            if filename is None:
                # Generate filename from URL hash
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"{url_hash}.png"
            
            filepath = self.images_dir / filename
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filename}")
            return str(filepath.relative_to(self.dataset_dir))
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    
    def add_entry(self, image_path, formula, description, tags=None, parameters=None, 
                  resolution=None, color_scheme=None):
        """Add a new entry to the dataset metadata"""
        
        # Validate image exists
        if not (self.dataset_dir / image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return False
        
        # Get image resolution if not provided
        if resolution is None:
            try:
                with Image.open(self.dataset_dir / image_path) as img:
                    resolution = list(img.size)
            except Exception as e:
                logger.warning(f"Could not get resolution for {image_path}: {e}")
                resolution = [512, 512]  # Default
        
        entry = {
            "image_path": image_path,
            "formula": formula,
            "description": description,
            "tags": tags or [],
            "parameters": parameters or {},
            "resolution": resolution,
            "color_scheme": color_scheme or "viridis"
        }
        
        # Append to metadata file
        with open(self.metadata_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Added entry: {description}")
        return True
    
    def ingest_sample_data(self):
        """Create some sample mathematical art entries"""
        logger.info("Creating sample mathematical art entries...")
        
        # Sample entries (these would normally be downloaded from actual sources)
        sample_entries = [
            {
                "formula": "r = a * sin(n * theta)",
                "description": "Rose curve with n=3",
                "tags": ["rose", "polar", "trigonometric"],
                "parameters": {"a": 1.0, "n": 3},
                "color_scheme": "viridis"
            },
            {
                "formula": "x = t * cos(t), y = t * sin(t)",
                "description": "Archimedean spiral",
                "tags": ["spiral", "parametric", "growth"],
                "parameters": {"t_range": [0, 20]},
                "color_scheme": "plasma"
            },
            {
                "formula": "z = x^2 + y^2",
                "description": "Paraboloid surface",
                "tags": ["3d", "surface", "quadratic"],
                "parameters": {"x_range": [-2, 2], "y_range": [-2, 2]},
                "color_scheme": "coolwarm"
            }
        ]
        
        for i, entry in enumerate(sample_entries, 1):
            # In a real implementation, you would generate or download the actual image
            image_path = f"images/{i:04d}.png"
            logger.info(f"Would create/download image: {image_path}")
            
            # Create placeholder entry
            self.add_entry(
                image_path=image_path,
                formula=entry["formula"],
                description=entry["description"],
                tags=entry["tags"],
                parameters=entry["parameters"],
                resolution=[512, 512],
                color_scheme=entry["color_scheme"]
            )

def main():
    """Main ingestion process"""
    logger.info("Starting mathematical art ingestion...")
    
    ingestor = MathArtIngestor()
    
    # For now, create sample data
    # In a real implementation, this would:
    # 1. Connect to math art repositories
    # 2. Download images and extract metadata
    # 3. Process and validate the data
    # 4. Store in the dataset structure
    
    ingestor.ingest_sample_data()
    
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    main()
