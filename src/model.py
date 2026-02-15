import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional
import config

class PlantDiseaseClassifier(nn.Module):
    """Классификатор заболеваний растений"""
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, model_name: str = config.MODEL_NAME):
        super(PlantDiseaseClassifier, self).__init__()
        
        self.model_name = model_name
        
        if 'efficientnet' in model_name:
            # EfficientNet
            self.backbone = timm.create_model(model_name, pretrained=True)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Linear(in_features, num_classes)
            
        elif 'resnet' in model_name:
            # ResNet
            self.backbone = timm.create_model(model_name, pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Linear(in_features, num_classes)
            
        elif 'vit' in model_name:
            # Vision Transformer
            self.backbone = timm.create_model(model_name, pretrained=True)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            self.classifier = nn.Linear(in_features, num_classes)
            
        else:
            raise ValueError(f"Неподдерживаемая модель: {model_name}")
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def get_model(device: torch.device) -> nn.Module:
    """Получение модели и перенос на устройство"""
    
    model = PlantDiseaseClassifier()
    
    # Используем DataParallel если несколько GPU
    if torch.cuda.device_count() > 1:
        print(f"Используем {torch.cuda.device_count()} GPU")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    return model