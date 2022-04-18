import sys; sys.path.append('..') # noqa
import os
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

from gerumo.data.dataset import describe_dataset
from gerumo.data.generators import build_generator
from gerumo.models.base import build_model
from gerumo.utils.engine import (
    setup_cfg, setup_environment, setup_experiment, setup_model,
    build_dataset, build_callbacks, build_metrics, build_optimizer, build_loss
)
from gerumo.visualization.metrics import training_history


def main(args):
    # Setup
    cfg = setup_cfg(args)
    output_dir = setup_experiment(cfg)
    logger = setup_environment(cfg)
    # Load datasets
    train_dataset = build_dataset(cfg, 'train')
    describe_dataset(train_dataset, logger,
                     save_to=output_dir / 'train_description.txt')
    validation_dataset = build_dataset(cfg, 'validation')
    describe_dataset(validation_dataset, logger,
                     save_to=output_dir / 'validation_description.txt')
    # Setup Generators
    train_generator = build_generator(cfg, train_dataset)
    validation_generator = build_generator(cfg, validation_dataset)
    # Build model
    input_shape = train_generator.get_input_shape()
    model = build_model(cfg, input_shape)
    output_dim = model.get_output_dim()
    # Build solver tools
    callbacks = build_callbacks(cfg)
    metrics = build_metrics(cfg, model=model)
    optimizer = build_optimizer(cfg, len(train_generator))
    loss = build_loss(cfg, output_dim)
    # Compile model
    model = setup_model(model, train_generator, optimizer, loss, metrics)
    # Star Training
    train_generator.fit_mode()
    validation_generator.fit_mode()
    model.fit_mode()
    start_time = time.time()
    history = model.fit(
        train_generator,
        epochs=cfg.SOLVER.EPOCHS,
        verbose=1 if args.verbose else 2,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        use_multiprocessing=False,
        workers=1,
        max_queue_size=20,
    )
    training_time = (time.time() - start_time) / 60.0
    logger.info(f'Training time: {training_time:.3f} [min]')
    training_history(history, training_time, cfg.EXPERIMENT_NAME,
                     save_to=output_dir)
    training_history(history, training_time, cfg.EXPERIMENT_NAME, ylog=True,
                     save_to=output_dir)
    return history, model, output_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a neural network model.')
    parser.add_argument('--config-file', required=True, metavar='FILE',
                        help='Path to config file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose training.')
    parser.add_argument(
        'opts',
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated 'PATH.KEY VALUE' pairs.
For python-based LazyConfig, use 'path.key=value'.
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    history, model, output_dir = main(args)
    print(output_dir)
