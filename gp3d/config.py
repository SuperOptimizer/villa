VESUVIUS_ROOT = "/vesuvius/"
ZARRS_PATH = f'{VESUVIUS_ROOT}/fragments/'
FRAGMENT_MASKS_PATH = f'{VESUVIUS_ROOT}/train_scrolls/'
INK_LABELS_PATH = f'{VESUVIUS_ROOT}/inklabels/'
OUTPUT_PATH = f'{VESUVIUS_ROOT}/inkdet_outputs/'

CHUNK_SIZE = 64
STRIDE = 32
ISO_THRESHOLD = 64
OUTPUT_SIZE = 4
BATCH_SIZE = 64
NUM_WORKERS = 12
NUM_EPOCHS = 2000
LEARNING_RATE = 8e-4
MIN_LEARNING_RATE = 8e-7
WEIGHT_DECAY = 5e-3
SEED = 42
VALIDATION_SPLIT = 0.0
INKDETECT_MEAN = .05
AUGMENT = True
RESNET_DEPTH = 50
AUGMENT_CHANCE = 0.5
GRADIENT_ACCUMULATE=4

