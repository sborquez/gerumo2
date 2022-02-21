import sys; sys.path.append('..') # noqa
import os
import time
import pickle

from gerumo.data.dataset import describe_dataset
from gerumo.data.generators import build_generator
from gerumo.utils.engine import (
    setup_cfg, setup_environment, setup_experiment, build_dataset, build_metrics
)
from gerumo.models.base import build_model
from gerumo.utils.structures import Event


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
    print('Training batches:', len(train_generator))
    print('Validation batches:', len(validation_generator))
    # Build model
    input_shape = train_generator.get_input_shape()
    model = build_model(cfg, input_shape)
    # Build metrics
    metrics = build_metrics(cfg, standalone=True)
    # Star Training
    inputs_batch, outputs_batch = train_generator.get_batch()
    start_time = time.time()
    model.fit(inputs_batch, outputs_batch)
    training_time = (time.time() - start_time) / 60.0
    logger.info(f'Training time: {training_time:.3f} [min]')
    # Validation
    start_time = time.time()
    inputs_batch, outputs_batch = validation_generator.get_batch()
    predicted_batch = model(inputs_batch)
    validation_time = (time.time() - start_time) / 60.0
    logger.info(f'Validation time: {validation_time:.3f} [min]')
    for name, metric in metrics.items():
        logger.info(f'{name}:\t{metric(predicted_batch, Event.list_to_tensor(outputs_batch)):.2f}')  # noqa
    # Save model
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    logger.info('Saved "model.pkl"')
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config-file', required=True, metavar='FILE',
                        help='path to config file')
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
    main(args)
