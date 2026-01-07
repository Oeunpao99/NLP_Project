#!/usr/bin/env python3
"""
Download or prepare model files for Khmer NER
"""

import os
import json
import shutil
from pathlib import Path

def prepare_model_files():
    """Prepare model files and label mappings"""
    
    # Create directories
    model_dir = Path("ml/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create label mappings (you need to adapt this based on your actual labels)
    label2idx = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8
    }
    
    idx2label = {i: l for l, i in label2idx.items()}
    
    # Save label mappings
    label_data = {
        "label2idx": label2idx,
        "idx2label": idx2label
    }
    
    with open(model_dir / "label_mappings.json", "w", encoding="utf-8") as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created label mappings at {model_dir}/label_mappings.json")
    print("\nNext steps:")
    print("1. Copy your model files to ml/model/ directory:")
    print("   - bilstm_crf_best.pt (NER model)")
    print("   - char_autoencoder_epoch3.pt (character autoencoder)")
    print("\n2. Update label_mappings.json with your actual label set")
    print("\n3. Run: docker-compose up --build")

if __name__ == "__main__":
    prepare_model_files()