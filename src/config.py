import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Dataset Settings
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2
BATCH_SIZE = 32

# Training
EPOCHS = 10
LEARNING_RATE = 1e-4

# Set these after verifying kagglehub download
RAW_DATA_PATH = os.path.join(DATA_DIR, "breast_histopathology_images")


