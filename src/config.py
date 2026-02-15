import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path("C:/Users/kozac/OneDrive/Рабочий стол/Нейросетевые технологии/plants_disease_classes")
DATA_DIR = BASE_DIR / "data"
PLANTVILLAGE_DIR = DATA_DIR / "plantvillage" / "raw" / "color"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Создание директорий
for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR, PLANTVILLAGE_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Параметры модели
MODEL_NAME = "efficientnet_b0"  # или "resnet50", "vit_base_patch16_224"
NUM_CLASSES = 2  # Бинарная классификация (здоровое/больное)
INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Параметры данных
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Параметры аугментации
AUGMENTATION = {
    'rotation': 20,
    'horizontal_flip': True,
    'vertical_flip': False,
    'brightness': 0.2,
    'contrast': 0.2,
}

# Пути для сохранения
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
LAST_MODEL_PATH = MODELS_DIR / "last_model.pth"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
METRICS_PATH = REPORTS_DIR / "metrics.txt"
TRAINING_CURVES_PATH = REPORTS_DIR / "training_curves.png"

# Параметры устройства
DEVICE = "cuda"  # Используем GPU