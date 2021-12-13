import os
import sys
import random
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, metrics, optimizers, losses
from ..config.config import CfgNode, get_cfg
from ..utils.structures import Task
from ..data.dataset import load_dataset, aggregate_dataset


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def setup_environment(cfg):
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

    return logging.getLogger("[GERUMO]")


def setup_experiment(cfg: CfgNode) -> Path:
    # Experiment folder
    cfg.defrost()
    output_dir = Path(cfg.OUTPUT_DIR).absolute()
    experiment_folder = f'{cfg.EXPERIMENT_NAME}'.replace(' ', '_').lower()
    experiment_folder += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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


def overwrite_output_dir(cfg: CfgNode):
    """"Add custom output dir using configuratioCNn options"""
    cfg.defrost()
    model_folder = f'freeze_at_{str(cfg.MODEL.BACKBONE.FREEZE_AT)}_'
    model_folder += f'resize_{cfg.INPUT.MAX_SIZE_TRAIN}_{cfg.INPUT.MIN_SIZE_TRAIN[0]}'
    model_folder += f'_anchors_{len(cfg.MODEL.ANCHOR_GENERATOR.SIZES[0])}'
    model_folder += f'_aspect_ratios_{len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0])}'
    model_folder += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    backbone = '_'.join(cfg.MODEL.BACKBONE.NAME.split('_')[1:-1])
    output_dir = Path(cfg.OUTPUT_DIR).absolute() / backbone.lower() / model_folder
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()


def build_dataset(cfg: CfgNode, subset: str):
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
    az = cfg.DATASETS.AGGREGATION.CENTER_AZ
    log10_mc_energy = cfg.DATASETS.AGGREGATION.LOG10_ENERGY
    hdf5_file = cfg.DATASETS.AGGREGATION.HDF5_FILEPATH
    remove_nan = cfg.DATASETS.AGGREGATION.REMOVE_NAN
    dataset = aggregate_dataset(dataset,
                                az, log10_mc_energy, hdf5_file, remove_nan)
    return dataset


def build_callbacks(cfg: CfgNode) -> List[callbacks.Callback]:
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
    if cfg.CALLBACKS.TENSORBOARD.ENABLE:
        callbacks_.append(
            callbacks.TensorBoard(
                log_dir=Path(cfg.OUTPUT_DIR) / 'logs'
            )
        )
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
    if cfg.CALLBACKS.CSVLOGGER.ENABLE:
        callbacks_.append(
            callbacks.CSVLogger(
                filename=Path(cfg.OUTPUT_DIR) / 'results.csv'
            )
        )
    return callbacks_


def build_metrics(cfg: CfgNode) -> List[metrics.Metric]:
    metrics_ = []
    if Task[cfg.MODEL.TASK] is Task.REGRESSION:
        for metric in cfg.METRICS.REGRESSION:
            metrics_.append(metric)
    elif Task[cfg.MODEL.TASK] is Task.CLASSIFICATION:
        for metric in cfg.METRICS.CLASSIFICATION:
            if 'Metric' == 'PRC':
                metric = metrics.AUC('prc', curve='PR')
            metrics_.append(metric)
    return metrics_


def build_optimizer(cfg: CfgNode) -> Any:
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


def build_loss(cfg: CfgNode) -> Any:
    loss_config = {k: v for k, v in cfg.SOLVER.LOSS.CONFIG}
    loss_ = losses.get({
        'class_name': cfg.SOLVER.LOSS.CLASS_NAME,
        'config': loss_config
    })
    return loss_
