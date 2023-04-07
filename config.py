import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = 'fc'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_WORKERS = 2 if DEVICE == 'cuda' else 1
DROP_PROBABILITY = 0.5