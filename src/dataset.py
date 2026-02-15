import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pathlib import Path
import random
from typing import Tuple, List, Dict
import config

class PlantDataset(Dataset):
    """Датасет для классификации растений"""
    
    def __init__(self, root_dir: Path, transform=None, class_names=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Определяем классы (здоровые/больные)
        if class_names is None:
            # Для PlantVillage: предполагаем структуру папок по заболеваниям
            # Здоровые: классы содержащие "healthy"
            # Больные: все остальные
            self.class_to_idx = {}
            classes = sorted([d for d in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, d))])
            
            for idx, class_name in enumerate(classes):
                # Определяем бинарную метку: 0 - здоровый, 1 - больной
                if 'healthy' in class_name.lower():
                    self.class_to_idx[class_name] = 0
                else:
                    self.class_to_idx[class_name] = 1
        else:
            self.class_to_idx = class_names
        
        # Загружаем изображения
        for class_name, label in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
        
        print(f"Загружено {len(self.images)} изображений")
        print(f"Распределение классов: здоровых={self.labels.count(0)}, больных={self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Загрузка изображения
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """Получение трансформаций для обучения и валидации"""
    
    # Трансформации для обучения с аугментацией
    train_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip() if config.AUGMENTATION['horizontal_flip'] else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if config.AUGMENTATION['vertical_flip'] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(config.AUGMENTATION['rotation']),
        transforms.ColorJitter(
            brightness=config.AUGMENTATION['brightness'],
            contrast=config.AUGMENTATION['contrast']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформации для валидации/тестирования (без аугментации)
    val_transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_dataloaders(root_dir: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создание загрузчиков данных для train/val/test"""
    
    train_transform, val_transform = get_transforms()
    
    # Загружаем полный датасет
    full_dataset = PlantDataset(root_dir, transform=train_transform)
    
    # Разделяем на train/val/test
    train_size = int(config.TRAIN_SPLIT * len(full_dataset))
    val_size = int(config.VAL_SPLIT * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Применяем разные трансформации
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Создаем загрузчики
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Размеры выборок: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader