import os
import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, List
from config import *


def initialize_vgg19(num_classes: int = NUM_CLASSES, feature_extract: bool = True, checkpoint_path: str = None) -> Tuple[nn.Module, List]:
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            # Handle both old format (state_dict only) and new format (dict with model_state_dict)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Old format - direct state_dict
                model.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            print(f"Checkpoint file size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
            print("Falling back to ImageNet pretrained weights")
    
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    
    model = model.to(DEVICE)
    
    return model, params_to_update


def load_model(model_path: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    model, _ = initialize_vgg19(num_classes=num_classes, feature_extract=True)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    # Handle both old format (state_dict only) and new format (dict with model_state_dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Old format - direct state_dict
        model.load_state_dict(checkpoint)
    model.eval()
    return model

