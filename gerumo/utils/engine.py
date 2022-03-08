import os
import sys
import random
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Any, Mapping, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, metrics, optimizers

from ..config.config import CfgNode, get_cfg
from ..utils.structures import Task
from ..data.dataset import load_dataset, aggregate_dataset
from ..models.losses import build_loss
from gerumo.models.base import build_model

build_model = build_model
build_loss = build_loss


def get_dataset_name(cfg, subset):
    if subset == 'test':
        dataset = cfg.DATASETS.TEST.EVENTS
    elif (subset == 'val') or (subset == 'validation'):
        dataset = cfg.DATASETS.VALIDATION.EVENTS
    else:
        dataset = cfg.DATASETS.TRAIN.EVENTS
    return '_'.join(Path(dataset).name.split('_')[:-1])


def setup_cfg(args):
    """Load configuration object from cmd arguments."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def setup_environment(cfg):
    """Setup device and setup a seed for the environment"""
    # Disable cuda
    if cfg.MODEL.DEVICE == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    # Setup random
    if cfg.SEED != -1:
        seed = cfg.SEED
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
    if cfg.DETERMINISTIC:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    # Change root logger level from WARNING (default) to NOTSET in order for
    # all messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)

    # Add stdout handler, with level INFO
    formatter = logging.Formatter('%(name)s: %(levelname)-8s %(message)s')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    # Add file rotating handler, with level INFO
    fileHandler = logging.FileHandler(Path(cfg.OUTPUT_DIR) / 'run.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logging.getLogger().addHandler(fileHandler)

    return logging.getLogger('[GERUMO]')


def setup_experiment(cfg: CfgNode, training=True) -> Path:
    """Setup experiment folders for training or evaluation a model."""
    # Experiment folder
    output_dir = Path(cfg.OUTPUT_DIR).absolute()
    # Evaluation
    if not training:
        if not output_dir.exists:
            raise ValueError(f'{output_dir} does not exist.')
        evaluation_dir = output_dir / 'evaluation'
        evaluation_dir.mkdir(exist_ok=True)
        return output_dir, evaluation_dir
    else:
        # Training
        cfg.defrost()
        experiment_folder = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_'
        experiment_folder += f'{cfg.EXPERIMENT_NAME}'.replace(' ', '_').lower()
        new_output_dir = output_dir / experiment_folder
        cfg.OUTPUT_DIR = str(new_output_dir)
        cfg.freeze()
        # Create new folder
        new_output_dir.mkdir(parents=True, exist_ok=False)
        # Copy config
        with open(new_output_dir / 'config.yml', 'w') as f:
            f.write(cfg.dump())
    return new_output_dir


def setup_model(model, generator, optimizer, loss, metrics):
    """Build models defining layers shapes and compile it."""
    X = generator[0][0]
    _ = model(X)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    model.summary()
    return model


def load_model(model, generator: Any, output_dir: Union[Path, str], epoch_idx: int = -1):
    """Load model's weights from a checkpoint at given epoch."""
    checkpoints = sorted(list(Path(output_dir).glob('weights/*.h5')))
    if len(checkpoints) == 0:
        raise ValueError('Model doesn`t have checkpoints')
    if epoch_idx >= 0 and len(checkpoints) <= epoch_idx:
        raise ValueError(f'Epoch_idx should be in [0, {len(checkpoints)}) or -1: {epoch_idx}')
    checkpoint = checkpoints[epoch_idx]
    X = generator[0][0]
    _ = model(X)
    try:
        model.load_weights(checkpoint)
    except ValueError:
        model._get_model()
        model.load_weights(checkpoint)
    return model


def build_dataset(cfg: CfgNode, subset: str):
    """Load a dataset subset from configuration."""
    # Load dataset subset
    if subset == 'train':
        events_path = cfg.DATASETS.TRAIN.EVENTS
        telescopes_path = cfg.DATASETS.TRAIN.TELESCOPES
        replace_folder = cfg.DATASETS.TRAIN.FOLDER
    elif subset == 'validation':
        events_path = cfg.DATASETS.VALIDATION.EVENTS
        telescopes_path = cfg.DATASETS.VALIDATION.TELESCOPES
        replace_folder = cfg.DATASETS.VALIDATION.FOLDER
    elif subset == 'test':
        events_path = cfg.DATASETS.TEST.EVENTS
        telescopes_path = cfg.DATASETS.TEST.TELESCOPES
        replace_folder = cfg.DATASETS.TEST.FOLDER
    else:
        raise ValueError('Invalid subset', subset)
    dataset = load_dataset(events_path, telescopes_path, replace_folder)
    # Aggregate
    center_az = cfg.DATASETS.AGGREGATION.CENTER_AZ
    log10_mc_energy = cfg.DATASETS.AGGREGATION.LOG10_ENERGY
    hdf5_file = cfg.DATASETS.AGGREGATION.HDF5_FILEPATH
    remove_nan = cfg.DATASETS.AGGREGATION.REMOVE_NAN
    ignore_particle_types = cfg.DATASETS.AGGREGATION.IGNORE_PARTICLE_TYPES
    if cfg.DATASETS.AGGREGATION.IGNORE_BY_DOMAINS:
        domains = {
            k: v for (k, v) in zip(
                cfg.OUTPUT.REGRESSION.TARGETS,
                cfg.OUTPUT.REGRESSION.TARGETS_DOMAINS
            )
        }
    else:
        domains = None
    dataset = aggregate_dataset(dataset, center_az, log10_mc_energy, hdf5_file,
                                remove_nan, ignore_particle_types, domains)
    return dataset


def build_callbacks(cfg: CfgNode) -> List[callbacks.Callback]:
    """Setup callback for training Neural Networks from configuration."""
    callbacks_ = []
    # Early Stop
    if cfg.CALLBACKS.EARLY_STOP.ENABLE:
        callbacks_.append(
            callbacks.EarlyStopping(
                monitor=cfg.CALLBACKS.EARLY_STOP.MONITOR,
                min_delta=cfg.CALLBACKS.EARLY_STOP.MIN_DELTA,
                patience=cfg.CALLBACKS.EARLY_STOP.PATIENCE,
                verbose=1,
                restore_best_weights=cfg.CALLBACKS.EARLY_STOP.RESTORE_BEST_WEIGHTS  # noqa
            )
        )
    # Tensorboard logs
    if cfg.CALLBACKS.TENSORBOARD.ENABLE:
        callbacks_.append(
            callbacks.TensorBoard(
                log_dir=Path(cfg.OUTPUT_DIR) / 'logs'
            )
        )
    # Periodic checkpoint weights
    if cfg.CALLBACKS.MODELCHECKPOINT.ENABLE:
        folder_path = Path(cfg.OUTPUT_DIR) / 'weights'
        folder_path.mkdir()
        callbacks_.append(
            callbacks.ModelCheckpoint(
                filepath=folder_path / 'model.{epoch:02d}-{val_loss:.2f}.h5',  # noqa
                monitor=cfg.CALLBACKS.MODELCHECKPOINT.MONITOR,
                verbose=1,
                save_best_only=cfg.CALLBACKS.MODELCHECKPOINT.BEST_ONLY,
                save_weights_only=cfg.CALLBACKS.MODELCHECKPOINT.WEIGHTS_ONLY
            )
        )
    # Log metrics into a file
    if cfg.CALLBACKS.CSVLOGGER.ENABLE:
        callbacks_.append(
            callbacks.CSVLogger(
                filename=Path(cfg.OUTPUT_DIR) / 'results.csv'
            )
        )
    return callbacks_


def build_metrics(cfg: CfgNode, standalone=False) -> Union[List[Union[metrics.Metric, str]], Mapping[str, metrics.Metric]]:
    """Build keras metrics from configuration."""
    metrics_ = []
    names_ = []
    if Task[cfg.MODEL.TASK] is Task.REGRESSION:
        metric_list = cfg.METRICS.REGRESSION
    elif Task[cfg.MODEL.TASK] is Task.CLASSIFICATION:
        metric_list = cfg.METRICS.CLASSIFICATION
    for metric in metric_list:
        names_.append(metric)
        if metric == 'PRC':
            metric = metrics.AUC(curve='PR')
        metrics_.append(metric)
    if standalone:
        return {n: metrics.get(m) for (n, m) in zip(names_, metrics_)}
    return metrics_


def build_optimizer(cfg: CfgNode) -> Any:
    """Build optimizers for training Neural Networks from configuration."""
    if cfg.SOLVER.LR_EXPDECAY.ENABLE:
        lr_scheduler = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg.SOLVER.BASE_LR,
            decay_steps=cfg.SOLVER.LR_EXPDECAY.DECAY_STEPS,
            decay_rate=cfg.SOLVER.LR_EXPDECAY.DECAY_RATE,
            staircase=cfg.SOLVER.LR_EXPDECAY.STAIRCASE
        )
    else:
        lr_scheduler = cfg.SOLVER.BASE_LR

    optimizer_config = {k: v for k, v in cfg.SOLVER.OPTIMIZER.CONFIG}
    optimizer_config['learning_rate'] = lr_scheduler
    optimizer_ = optimizers.get({
        'class_name': cfg.SOLVER.OPTIMIZER.CLASS_NAME,
        'config': optimizer_config
    })
    return optimizer_
