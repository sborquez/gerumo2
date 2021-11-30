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
_C.MODEL.TELESCOPES = ["LST_LSTCam"]   # Telescope {type}_{camera_type}
_C.MODEL.WEIGHTS = None  # Path  to a checkpoint file to be loaded to the model
# Models Loss
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.NAME = "MAE"
_C.MODEL.LOSS.ARGS = []   # [Any, ...]
_C.MODEL.LOSS.KWARGS = []  # [(str, Any), ...]
# Model Architecture
_C.MODEL.ARCHITECTURE = CN()
_C.MODEL.ARCHITECTURE.NAME = "CNN"
_C.MODEL.ARCHITECTURE.ARGS = []   # [Any, ...]
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
_C.INPUT.MAPPER.ARGS = []   # [Any, ...]
_C.INPUT.MAPPER.KWARGS = []  # [(str, Any), ...]
_C.INPUT.IMAGE_CHANNELS = ["charge", "time_peaks", "mask"]
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
_C.OUTPUT.CLASSIFICATION.CLASSES = ["gamma", "protron"]
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
# Solver for Mono training
# ----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.MAX_ITER = 200
_C.SOLVER.METHOD = "Adam"
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.NESTEROV = False
_C.SOLVER.MOMENTUM = 0.9
# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------
_C.CALLBACKS = CN()
_C.CALLBACKS.EARLY_STOP_PATIENCE = 20
_C.CALLBACKS.CHECKPOINTS_PERIOD = 10
# ----------------------------------------------------------------------------
# Monitor
# ----------------------------------------------------------------------------
_C.MONITOR = CN()
_C.MONITOR.SAVE_LOSS = True
_C.MONITOR.VIS_PERIOD = 0
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
