import os
import shutil
from pathlib import Path
from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
from config import *


def split_dataset(source_dir: str, target_dir: str, train_split: float = 0.8, val_split: float = 0.2, seed: int = 42) -> None:
    if os.path.exists(target_dir):
        print(f"Dataset directory {target_dir} already exists. Skipping split.")
        return
    
    os.makedirs(target_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)
    
    np.random.seed(seed)
    
    for class_name in CLASS_NAMES:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_path} not found")
            continue
        
        all_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_files = np.array(all_files)
        
        if len(all_files) == 0:
            print(f"Warning: No images found in {class_path}")
            continue
        
        test_size = 1 - train_split
        train_val_files, test_files = train_test_split(
            all_files, test_size=test_size, random_state=seed, shuffle=True
        )
        
        val_size_of_trainval = val_split
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_size_of_trainval, random_state=seed, shuffle=True
        )
        
        for filename in train_files:
            src = os.path.join(class_path, filename)
            dst = os.path.join(target_dir, 'train', class_name, filename)
            shutil.copy2(src, dst)
        
        for filename in val_files:
            src = os.path.join(class_path, filename)
            dst = os.path.join(target_dir, 'val', class_name, filename)
            shutil.copy2(src, dst)
        
        for filename in test_files:
            src = os.path.join(class_path, filename)
            dst = os.path.join(target_dir, 'test', class_name, filename)
            shutil.copy2(src, dst)
        
        print(f"Class {class_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


def get_data_transforms() -> Dict[str, transforms.Compose]:
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=20,
                translate=None,
                scale=(0.8, 1.0),
                shear=20
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
    }
    return data_transforms


def get_dataloaders(dataset_dir: str, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS) -> Tuple[Dict[str, DataLoader], Dict[str, int], list]:
    data_transforms = get_data_transforms()
    
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

