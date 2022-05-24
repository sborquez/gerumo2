import os
import sys
import random
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Any, Mapping, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import callbacks, metrics, optimizers

from ..config.config import CfgNode, get_cfg
from ..data.dataset import load_dataset, aggregate_dataset
from ..data.generators import BaseGenerator
from ..models.base import build_model, BaseModel
from ..models.losses import build_loss
from ..utils.structures import Task


build_model = build_model
build_loss = build_loss


def get_dataset_name(cfg: CfgNode, subset: str) -> str:
    if subset == 'test':
        dataset = cfg.DATASETS.TEST.EVENTS
    elif (subset == 'val') or (subset == 'validation'):
        dataset = cfg.DATASETS.VALIDATION.EVENTS
    else:
        dataset = cfg.DATASETS.TRAIN.EVENTS
    return '_'.join(Path(dataset).name.split('_')[:-1])


def setup_cfg(args) -> CfgNode:
    """Load configuration object from cmd arguments."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def setup_environment(cfg: CfgNode) -> logging.Logger:
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


def setup_experiment(cfg: CfgNode, training: bool = True, ensemble: bool=False) -> Path:
    """Setup experiment folders for training or evaluation a model."""
    # Experiment folder
    output_dir = Path(cfg.OUTPUT_DIR).absolute()
    # Evaluation
    if ensemble:
        output_dir.mkdir(exist_ok=True)
        return output_dir
    elif not training:
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


def setup_model(model: BaseModel, generator: BaseGenerator, optimizer: optimizers.Optimizer,
                loss: Any, metrics: list) -> BaseModel:
    """Build models defining layers shapes and compile it."""
    X = generator[0][0]
    _ = model(X)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    try:
        model.summary()
    except Exception:
        print('Summary not available')
    return model


def load_model(model: BaseModel, generator: BaseGenerator, output_dir: Union[Path, str], epoch_idx: int = -1):
    """Load model's weights from a checkpoint at given epoch."""
    if os.path.isdir(output_dir):
        checkpoints = sorted(list(Path(output_dir).glob('weights/*.h5')))
        if len(checkpoints) == 0:
            raise ValueError('Model doesn`t have checkpoints')
        if epoch_idx >= 0 and len(checkpoints) <= epoch_idx:
            raise ValueError(f'Epoch_idx should be in [0, {len(checkpoints)}) or -1: {epoch_idx}')
        checkpoint = checkpoints[epoch_idx]
    elif os.path.isfile(output_dir) and output_dir.endswith('.h5'):
        checkpoint = output_dir
    else:
        raise ValueError('output_dir is not a folder or a checkpoint', output_dir)
    X = generator[0][0]
    _ = model(X)
    try:
        model.load_weights(checkpoint)
    except ValueError:
        model._get_model()
        model.load_weights(checkpoint)
    return model


def build_dataset(cfg: CfgNode, subset: str) -> pd.DataFrame:
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
                log_dir=Path(cfg.OUTPUT_DIR) / 'logs',
                update_freq='batch',
                write_images=True,
                histogram_freq=1
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


class _MetricWrapper(tf.keras.metrics.Metric):
    def __init__(self, metric_name, model, *args, **kwargs) -> None:
        super().__init__(name=metric_name, *args, **kwargs)
        self.metric = tf.keras.metrics.get(metric_name)
        self.model = model

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_point = self.model.point_estimation(y_pred)
        self.metric.update_state(y_true, y_point, *args, **kwargs)

    def reset_state(self):
        self.metric.reset_state()
        
    def result(self):
        return self.metric.result()


def _point_estimation_wrapper(cfg, model):
    if cfg.MODEL.ARCHITECTURE.NAME in ('UmonneModel',):  # Custom wrapper for models with different output
        def get_wrapper(metric_name):
            return _MetricWrapper(metric_name, model)
        return get_wrapper
    else:
        return lambda metric_name: metric_name


def build_metrics(cfg: CfgNode, standalone: bool = False, model: Optional[BaseModel] = None) -> Union[List[Union[metrics.Metric, str]], Mapping[str, metrics.Metric]]:
    """Build keras metrics from configuration."""
    metrics_ = []
    names_ = []
    if Task[cfg.MODEL.TASK] is Task.REGRESSION:
        metric_list = cfg.METRICS.REGRESSION
        if model is not None:
            metric_list = map(_point_estimation_wrapper(cfg, model), metric_list)
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


def build_optimizer(cfg: CfgNode, steps_per_epoch: int = 1) -> optimizers.Optimizer:
    """Build optimizers for training Neural Networks from configuration."""
    if cfg.SOLVER.CYCLICAL_LR.ENABLE:
        scale_functions = {
            'linear': lambda x: 1,
            'fixed_decay': lambda x: 1 / (2.0**(tf.cast(x, tf.float32) - 1))
        }
        lr_scheduler = tfa.optimizers.CyclicalLearningRate(
            initial_learning_rate=cfg.SOLVER.BASE_LR,
            maximal_learning_rate=cfg.SOLVER.CYCLICAL_LR.MAX_LR,
            step_size=cfg.SOLVER.CYCLICAL_LR.FACTOR * steps_per_epoch,
            scale_fn=scale_functions[cfg.SOLVER.CYCLICAL_LR.SCALE_FN],
            scale_mode=cfg.SOLVER.CYCLICAL_LR.MODE
        )
    elif cfg.SOLVER.LR_EXPDECAY.ENABLE:
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


