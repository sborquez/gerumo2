from .config import CfgNode as CN
from ..utils.structures import Task, ReconstructionMode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
# -----------------------------------------------------------------------------
# Model Mono
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.RECONSTRUCTION_MODE = ReconstructionMode.SINGLE.name
_C.MODEL.TASK = Task.REGRESSION.name
_C.MODEL.TELESCOPES = ["LST"]   # Telescope type LST|MST|SST
_C.MODEL.WEIGHTS = None  # Path  to a checkpoint file to be loaded to the model
# Model Architecture
_C.MODEL.ARCHITECTURE = CN()
_C.MODEL.ARCHITECTURE.NAME = "CNN"
_C.MODEL.ARCHITECTURE.KWARGS = []  # [(str, Any), ...]
# -----------------------------------------------------------------------------
# Ensembler Stereo/Multi-Stereo
# -----------------------------------------------------------------------------
_C.ENSEMBLER = CN()
_C.ENSEMBLER.DEVICE = "cuda"
_C.ENSEMBLER.NAME = "IntensityWeightedAverage"
_C.ENSEMBLER.TELESCOPES = []    # Telescope {type}_{camera_type}
_C.ENSEMBLER.ARCHITECTURES = []  # Checkpoint with architecture files
_C.ENSEMBLER.WEIGHTS = []       # Only weights checkpoint files
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.POINTING = (0, 20)
# Train dataset paths to events and telescopes folder or parquet files
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.EVENTS = None
_C.DATASETS.TRAIN.TELESCOPES = None
_C.DATASETS.TRAIN.FOLDER = None
# Validation dataset paths to events and telescopes folder or parquet files
_C.DATASETS.VALIDATION = CN()
_C.DATASETS.VALIDATION.EVENTS = None
_C.DATASETS.VALIDATION.TELESCOPES = None
_C.DATASETS.VALIDATION.FOLDER = None
# Test dataset paths to events and telescopes folder or parquet files
_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.EVENTS = None
_C.DATASETS.TEST.TELESCOPES = None
_C.DATASETS.TEST.FOLDER = None
# Agregate dataset
_C.DATASETS.AGGREGATION = CN()
_C.DATASETS.AGGREGATION.CENTER_AZ = True
_C.DATASETS.AGGREGATION.LOG10_ENERGY = True
_C.DATASETS.AGGREGATION.HDF5_FILEPATH = True
_C.DATASETS.AGGREGATION.REMOVE_NAN = True
_C.DATASETS.AGGREGATION.IGNORE_PARTICLE_TYPES = []
# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------
_C.GENERATOR = CN()
# Number of data loading threads
_C.GENERATOR.NAME = "MonoGenerator"
_C.GENERATOR.NUM_WORKERS = 1
# Shuffle Dataset
_C.GENERATOR.ENABLE_SHUFFLE = True
# If shufle is enable, it keep same h5 file in a batch
_C.GENERATOR.USE_STRICT_SHUFFLE = False
# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.MAPPER = CN()
_C.INPUT.MAPPER.NAME = "SimpleSquareImage"
_C.INPUT.MAPPER.KWARGS = []  # [(str, Any), ...]
_C.INPUT.IMAGE_CHANNELS = ["image", "peak_time", "image_mask"]
_C.INPUT.TELESCOPE_FEATURES = []
# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.MAPPER = CN()
_C.OUTPUT.MAPPER.NAME = "SimpleRegression"
# -----------------------------------------------------------------------------
# Classification task
# -----------------------------------------------------------------------------
_C.OUTPUT.CLASSIFICATION = CN()
_C.OUTPUT.CLASSIFICATION.TARGET = "particle_type"
_C.OUTPUT.CLASSIFICATION.NUM_CLASSES = 2
_C.OUTPUT.CLASSIFICATION.CLASSES = ["gamma", "proton"]
# -----------------------------------------------------------------------------
# Regression task
# -----------------------------------------------------------------------------
_C.OUTPUT.REGRESSION = CN()
_C.OUTPUT.REGRESSION.TARGETS = ["true_az", "true_alt"]
_C.OUTPUT.REGRESSION.TARGETS_DOMAINS = [
    (1.15, 1.3),
    (-0.25, 0.25)
]
# ----------------------------------------------------------------------------
# Solver for training NN
# ----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.EPOCHS = 200
_C.SOLVER.BASE_LR = 0.001
# Optimizer for Neural Networks
_C.SOLVER.OPTIMIZER = CN()
_C.SOLVER.OPTIMIZER.CLASS_NAME = "RMSprop"
_C.SOLVER.OPTIMIZER.CONFIG = [
    ('rho', 0.9),
    ('momentum', 0.0),
    ('epsilon', 1e-07),
    ('centered', False)
]
# LR Scheduler Exponential Decay
_C.SOLVER.LR_EXPDECAY = CN()
_C.SOLVER.LR_EXPDECAY.ENABLE = True
_C.SOLVER.LR_EXPDECAY.DECAY_STEPS = 100000
_C.SOLVER.LR_EXPDECAY.DECAY_RATE = 0.96
_C.SOLVER.LR_EXPDECAY.STAIRCASE = True
# Training Batch size
# Models Loss
_C.SOLVER.LOSS = CN()
_C.SOLVER.LOSS.CLASS_NAME = "MeanAbsoluteError"  # Use loss class name
_C.SOLVER.LOSS.CONFIG = []  # [(str, Any), ...]
# ----------------------------------------------------------------------------
# Callbacks for NN training
# ----------------------------------------------------------------------------
_C.CALLBACKS = CN()
# Early Stop
_C.CALLBACKS.EARLY_STOP = CN()
_C.CALLBACKS.EARLY_STOP.ENABLE = True
_C.CALLBACKS.EARLY_STOP.PATIENCE = 20
_C.CALLBACKS.EARLY_STOP.MONITOR = 'val_loss'
_C.CALLBACKS.EARLY_STOP.MIN_DELTA = 0
_C.CALLBACKS.EARLY_STOP.RESTORE_BEST_WEIGHTS = False
# Tensorboard
_C.CALLBACKS.TENSORBOARD = CN()
_C.CALLBACKS.TENSORBOARD.ENABLE = True
# Model Checkpoint
_C.CALLBACKS.MODELCHECKPOINT = CN()
_C.CALLBACKS.MODELCHECKPOINT.ENABLE = True
_C.CALLBACKS.MODELCHECKPOINT.MONITOR = 'val_loss'
_C.CALLBACKS.MODELCHECKPOINT.BEST_ONLY = True
_C.CALLBACKS.MODELCHECKPOINT.WEIGHTS_ONLY = True
# CSVLog
_C.CALLBACKS.CSVLOGGER = CN()
_C.CALLBACKS.CSVLOGGER.ENABLE = True
# ----------------------------------------------------------------------------
# Metrics for training
# ----------------------------------------------------------------------------
_C.METRICS = CN()
# Use class names from this: https://keras.io/api/metrics/
_C.METRICS.CLASSIFICATION = [
    'SparseCategoricalAccuracy',
    'SparseCategoricalCrossentropy',
    'AUC',
    'PRC',
    'Precision',
    'Recall',
    'TruePositives',
    'TrueNegatives',
    'FalsePositives',
    'FalseNegatives'
]
_C.METRICS.REGRESSION = [
    'MeanSquaredError',
    'RootMeanSquaredError',
    'MeanAbsoluteError',
    'MeanAbsolutePercentageError',
    'MeanSquaredLogarithmicError',
    'CosineSimilarity',
    'LogCoshError'
]
# ----------------------------------------------------------------------------
# Misc options
# ----------------------------------------------------------------------------
_C.EXPERIMENT_NAME = None
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
_C.DETERMINISTIC = False
_C.VERSION = 1
