import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import config
from pathlib import Path
import json
from typing import Dict, List, Tuple

def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Расчет метрик классификации"""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
    }
    
    # Добавляем метрики для каждого класса
    class_report = classification_report(y_true, y_pred, output_dict=True)
    for class_name, class_metrics in class_report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics[f'class_{class_name}_f1'] = class_metrics['f1-score']
    
    return metrics

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: Path):
    """Построение и сохранение матрицы ошибок"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        train_accs: List[float], val_accs: List[float], save_path: Path):
    """Построение кривых обучения"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Потери
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Точность
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Val Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(metrics: Dict, save_path: Path):
    """Сохранение метрик в файл"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== Метрики классификации ===\n\n")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                f.write(f"{metric_name}: {metric_value:.4f}\n")
            else:
                f.write(f"{metric_name}: {metric_value}\n")

def visualize_predictions(model, test_loader, device, num_samples=9):
    """Визуализация предсказаний модели"""
    
    model.eval()
    images, labels, predictions = [], [], []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Сохраняем несколько примеров
            for i in range(min(len(data), num_samples - len(images))):
                images.append(data[i].cpu())
                labels.append(target[i].cpu())
                predictions.append(pred[i].cpu())
            
            if len(images) >= num_samples:
                break
    
    # Визуализация
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx in range(len(images)):
        img = images[idx].permute(1, 2, 0).numpy()
        # Денормализация
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        color = 'green' if labels[idx] == predictions[idx] else 'red'
        axes[idx].set_title(f'True: {labels[idx].item()}\nPred: {predictions[idx].item()}', 
                           color=color)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / 'predictions_visualization.png', dpi=300)
    plt.close()