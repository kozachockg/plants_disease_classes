import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

import config
from model import get_model
from dataset import create_dataloaders
from utils import calculate_metrics, plot_confusion_matrix, plot_training_curves, save_metrics, visualize_predictions

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Обучение одной эпохи"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Валидация одной эпохи"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss / len(val_loader), 100. * correct / total, all_preds, all_targets

def main():
    """Основная функция обучения"""
    
    print("=" * 50)
    print("Запуск обучения модели для классификации растений")
    print("=" * 50)
    
    # Определяем устройство
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Создаем загрузчики данных
    print("\nЗагрузка данных...")
    train_loader, val_loader, test_loader = create_dataloaders(config.PLANTVILLAGE_DIR)
    
    # Создаем модель
    print("\nСоздание модели...")
    model = get_model(device)
    
    # Определяем функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Для сохранения лучшей модели
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("\nНачало обучения...")
    for epoch in range(config.EPOCHS):
        print(f"\nЭпоха {epoch+1}/{config.EPOCHS}")
        
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Валидация
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Обновляем learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"✓ Сохранена лучшая модель с точностью {val_acc:.2f}%")
    
    print("\n" + "=" * 50)
    print("Обучение завершено!")
    print("=" * 50)
    
    # Загружаем лучшую модель для тестирования
    print("\nТестирование лучшей модели...")
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    
    # Тестирование
    test_loss, test_acc, test_preds, test_targets = validate_epoch(model, test_loader, criterion, device)
    
    # Расчет метрик
    metrics = calculate_metrics(test_targets, test_preds)
    metrics['test_loss'] = test_loss
    metrics['test_accuracy'] = test_acc
    
    print(f"\nРезультаты на тестовой выборке:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Визуализация
    print("\nСоздание визуализаций...")
    
    # Матрица ошибок
    plot_confusion_matrix(test_targets, test_preds, config.CONFUSION_MATRIX_PATH)
    print(f"✓ Матрица ошибок сохранена в {config.CONFUSION_MATRIX_PATH}")
    
    # Кривые обучения
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, config.TRAINING_CURVES_PATH)
    print(f"✓ Кривые обучения сохранены в {config.TRAINING_CURVES_PATH}")
    
    # Сохранение метрик
    save_metrics(metrics, config.METRICS_PATH)
    print(f"✓ Метрики сохранены в {config.METRICS_PATH}")
    
    # Визуализация предсказаний
    visualize_predictions(model, test_loader, device)
    print(f"✓ Визуализация предсказаний сохранена")
    
    print("\n" + "=" * 50)
    print("Проект успешно завершен!")
    print("=" * 50)

if __name__ == "__main__":
    main()