DATASET = "trash_images"
ROOT_DIR = r"dataset/trash_images/"
LR = 0.0003
SCHEDULER_MAX_IT = 30
WEIGH_DECAY = 1e-4
EPSILON = 1e-4

# train loop
BATCH_SIZE = 64
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
EPOCHS = 15
USE_INDEX = False
# callback
PATIENCE = 3

TOP_K_SAVES = 1
# training loop
NUM_TRIALS = 1

INDICES_DIR = "indices/"
CHECKPOINTS_DIR = "checkpoints/"
METRICS_DIR = "metrics/"
WANDB_PROJECT = "AI_BinBot_Project"

# model directories
CONVNEXT_DIR = CHECKPOINTS_DIR + "convnext/"

# model file names
CONVNEXT_FILENAME = "convnext_"

# csv file names
CONVNEXT_CSV_FILENAME = METRICS_DIR + CONVNEXT_FILENAME + "metrics.csv"
CONVNEXT_CSV_PER_CLASS_FILENAME = (
    METRICS_DIR + CONVNEXT_FILENAME + "per_class_metrics.csv"
)
CONVNEXT_CSV_CM_FILENAME = METRICS_DIR + CONVNEXT_FILENAME + "confusion_matrix.csv"
CONVNEXT_BILATERAL_CSV_FILENAME = (
    METRICS_DIR + CONVNEXT_FILENAME + "_bilateral_metrics.csv"
)

# transformations
ORIGINAL_SIZE = (299, 299)
RESIZE = 236
CROP = 224
STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]
ROTATION = 30

CLASS_NAMES = ["cardboard", "glass", "metal"]