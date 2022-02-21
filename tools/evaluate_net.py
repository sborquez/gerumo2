import sys; sys.path.append('..')  # noqa
import os
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'  # noqa

from gerumo.data.dataset import describe_dataset
from gerumo.data.generators import build_generator
from gerumo.models.base import build_model
from gerumo.utils.engine import (
    setup_cfg, setup_environment, setup_experiment, load_model, build_dataset
)


def main(args):
    # Setup
    cfg = setup_cfg(args)
    output_dir = setup_experiment(cfg, training=False)
    logger = setup_environment(cfg)
    # Load datasets
    evaluation_dataset_name = 'validation' if args.use_validation else 'test'
    evaluation_dataset = build_dataset(cfg, evaluation_dataset_name)
    describe_dataset(evaluation_dataset, logger,
                     save_to=output_dir / f'{evaluation_dataset_name}_description.txt')
    # Setup Generators
    evaluation_generator = build_generator(cfg, evaluation_dataset)
    # Load checkpoint model
    input_shape = evaluation_generator.get_input_shape()
    model = build_model(cfg, input_shape)
    model = load_model(model, evaluation_generator, output_dir, args.epoch)
    # Star Evaluation
    start_time = time.time()
    evaluation = None
    evaluation_time = (time.time() - start_time) / 60.0
    logger.info(f'Evaluation time: {evaluation_time:.3f} [min]')
    return evaluation, model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a trained neural network model.')
    parser.add_argument('--config-file', required=True, metavar='FILE',
                        help='path to config file')
    parser.add_argument('--epoch', default=-1, type=int,
                        help='Which epoch use to evaluate.')
    parser.add_argument('--use_validation', action='store_true',
                        help='Use "validation set" instead of "test set"')
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
    if os.path.isdir(args.config_file):
        args.config_file = os.path.join(args.config_file, 'config.yml')
    main(args)
