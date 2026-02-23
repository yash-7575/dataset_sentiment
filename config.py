"""
Configuration for ABSA (Aspect-Based Sentiment Analysis) project.
All hyperparameters, paths, and constants are defined here.
"""
import os
import torch

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR  # raw data lives alongside this file
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Training data (XML has the richest annotations)
TRAIN_FILES = {
    "laptops": os.path.join(DATA_DIR, "Laptop_Train_v2.xml"),
    "restaurants": os.path.join(DATA_DIR, "Restaurants_Train_v2.xml"),
}

# Test data (Phase B XML has aspect terms)
TEST_FILES = {
    "laptops": os.path.join(DATA_DIR, "Laptops_Test_Data_phaseB.xml"),
    "restaurants": os.path.join(DATA_DIR, "Restaurants_Test_Data_phaseB.xml"),
}

# ============================================================
# Label Mapping
# ============================================================
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2, "conflict": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)

# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# BERT Hyperparameters
# ============================================================
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 128
BERT_BATCH_SIZE = 16
BERT_LR = 2e-5
BERT_EPOCHS = 5
BERT_DROPOUT = 0.3
BERT_WARMUP_RATIO = 0.1

# ============================================================
# LSTM Hyperparameters
# ============================================================
LSTM_EMBED_DIM = 300
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.5
LSTM_BATCH_SIZE = 32
LSTM_LR = 1e-3
LSTM_EPOCHS = 15
LSTM_MAX_LEN = 100
GLOVE_PATH = os.path.join(BASE_DIR, "glove.6B.300d.txt")

# ============================================================
# Traditional ML Hyperparameters
# ============================================================
TFIDF_MAX_FEATURES = 10000
CONTEXT_WINDOW = 5  # words around aspect for feature extraction
RANDOM_STATE = 42

# ============================================================
# General
# ============================================================
VAL_SPLIT = 0.2
EARLY_STOP_PATIENCE = 3
SEED = 42
