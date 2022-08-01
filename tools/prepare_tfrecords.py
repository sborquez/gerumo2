import sys; sys.path.append('..') # noqa
import os
import time
from pathlib import Path

from gerumo.data.dataset import describe_dataset
from gerumo.data.tfrecords import generator_to_record
from gerumo.data.generators import build_generator
from gerumo.utils.engine import (
    setup_cfg, setup_environment, build_dataset
)


def main(args):
    # Setup
    cfg = setup_cfg(args)
    cfg.defrost()
    cfg.SOLVER.BATCH_SIZE = 1
    cfg.freeze()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples_per_file = args.samples_per_file
    logger = setup_environment(cfg)
    # Load datasets
    train_dataset = build_dataset(cfg, 'train')
    validation_dataset = build_dataset(cfg, 'validation')
    # Setup Generators
    train_generator = build_generator(cfg, train_dataset)
    validation_generator = build_generator(cfg, validation_dataset)
    input_shape = train_generator.get_input_shape()
    train_generator.fit_mode()
    validation_generator.fit_mode()
    # Convert dataset
    start_time = time.time()
    ## Training
    dataset_split = 'train'
    dataset_generator = train_generator
    describe_dataset(train_dataset, logger, save_to=output_dir / f'{dataset_split}_description.txt')
    generator_to_record(dataset_generator, dataset_split, output_dir, samples_per_file, input_shape)    
    ## Validation
    dataset_split = 'validation'
    dataset_generator = validation_generator
    describe_dataset(validation_dataset, logger, save_to=output_dir / f'{dataset_split}_description.txt')
    generator_to_record(dataset_generator, dataset_split, output_dir, samples_per_file, input_shape)
    convertion_time = (time.time() - start_time) / 60.0
    logger.info(f'Convertion time: {convertion_time:.3f} [min]')
    return output_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a neural network model.')
    parser.add_argument('--config-file', required=True, metavar='FILE',
                        help='Path to config file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose training.')
    parser.add_argument('-o', '--output_dir', required=True, metavar='DIR',
                        help='Output folder.')
    parser.add_argument('-s', '--samples_per_file', type=int, default=50000,
                        help='Samples for tf record.')    
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
    output_dir = main(args)
    print(output_dir)
