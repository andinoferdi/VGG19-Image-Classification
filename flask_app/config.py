import torch
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()
IMAGE_SIZE = 224
NUM_CLASSES = 6
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'all_data')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'vgg19_best.pth')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'flask_app', 'static', 'uploads')

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

