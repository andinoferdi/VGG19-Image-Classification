import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, Tuple, List
from config import *


def train_model(model: nn.Module, dataloaders: Dict, dataset_sizes: Dict, 
                criterion: nn.Module, optimizer: optim.Optimizer, 
                scheduler: optim.lr_scheduler._LRScheduler, num_epochs: int = EPOCHS) -> Tuple:
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 40)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history, time_elapsed


def evaluate_model(model: nn.Module, dataloader, class_names: List[str]) -> Tuple:
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    test_start = time.time()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_time = time.time() - test_start
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        roc_auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        idx = np.where(all_labels == i)[0]
        if len(idx) > 0:
            per_class_acc[class_name] = np.mean(all_preds[idx] == all_labels[idx])
        else:
            per_class_acc[class_name] = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'per_class_accuracy': per_class_acc,
        'test_time': test_time
    }
    
    return metrics


def save_plots(train_loss: List, val_loss: List, train_acc: List, val_acc: List, 
               cm: np.ndarray, class_names: List[str], save_dir: str = RESULTS_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(len(train_loss))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r-', label='Train Loss')
    plt.plot(epochs, val_loss, 'b-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'r-', label='Train Acc')
    plt.plot(epochs, val_acc, 'b-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=100, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar=True, square=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
    plt.close()


def save_metrics(metrics: Dict, train_time: float, save_dir: str = RESULTS_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Training Time: {train_time:.2f} seconds\n")
        f.write(f"Testing Time: {metrics['test_time']:.2f} seconds\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (macro): {metrics['precision']:.4f}\n")
        f.write(f"Recall (macro): {metrics['recall']:.4f}\n")
        f.write(f"F1-Score (macro): {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC (macro): {metrics['roc_auc']:.4f}\n\n")
        
        f.write("Per-Class Accuracy:\n")
        for class_name, acc in metrics['per_class_accuracy'].items():
            f.write(f"  {class_name}: {acc:.4f}\n")
        
        f.write(f"\n{metrics['classification_report']}\n")

