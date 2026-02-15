import torch
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

import config
from model import PlantDiseaseClassifier

def load_model(model_path: Path, device: torch.device):
    """Загрузка обученной модели"""
    
    model = PlantDiseaseClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path: Path, model, device: torch.device):
    """Предсказание для одного изображения"""
    
    # Трансформации для изображения
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка и предобработка изображения
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()

def main():
    """Основная функция для инференса"""
    
    # Определяем устройство
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    # Загружаем модель
    model_path = config.BEST_MODEL_PATH
    if not model_path.exists():
        print(f"Ошибка: Модель не найдена по пути {model_path}")
        return
    
    print(f"Загрузка модели из {model_path}...")
    model = load_model(model_path, device)
    print("Модель успешно загружена!")
    
    # Пример использования
    print("\n" + "=" * 50)
    print("Инференс на одном изображении")
    print("=" * 50)
    
    # Здесь можно указать путь к изображению
    # Для примера используем первое изображение из тестовой выборки
    test_image_path = config.PLANTVILLAGE_DIR / "test_image.jpg"
    
    if test_image_path.exists():
        class_idx, confidence, probs = predict_image(test_image_path, model, device)
        
        print(f"\nРезультаты для {test_image_path}:")
        print(f"Предсказанный класс: {class_idx} (0 - здоровый, 1 - больной)")
        print(f"Уверенность: {confidence:.2%}")
        print(f"\nВероятности по классам:")
        print(f"Здоровый: {probs[0]:.2%}")
        print(f"Больной: {probs[1]:.2%}")
    else:
        print(f"\nТестовое изображение не найдено по пути {test_image_path}")
        print("Укажите путь к изображению для предсказания")

if __name__ == "__main__":
    main()