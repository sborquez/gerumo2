import sys; sys.path.append('..')  # noqa
import os
import time

from tqdm import tqdm

from gerumo.data.dataset import describe_dataset
from gerumo.data.generators import build_generator
from gerumo.utils.engine import (
    setup_cfg, setup_environment, setup_experiment, build_dataset
)
from gerumo.utils.structures import Event, Task
from gerumo.models.base import build_ensembler
from gerumo.visualization import metrics


def main(args):
    # Setup
    # Find the config.yml
    if os.path.isdir(args.config_file):
        args.config_file = os.path.join(args.config_file, 'config.yml')
    # Load the configurations
    cfg = setup_cfg(args)
    evaluation_dir = setup_experiment(cfg, ensemble=True)
    logger = setup_environment(cfg)

    # Load evaluation dataset
    # Setup evaluation datasets directory
    if args.use_validation:
        evaluation_dataset_name = 'validation'
    else:
        evaluation_dataset_name = 'test'
    if args.dataset_name is not None:
        evaluation_subfolder = args.dataset_name
    else:
        evaluation_subfolder = evaluation_dataset_name
    evaluation_dir = evaluation_dir / evaluation_subfolder
    evaluation_dir.mkdir(exist_ok=True)
    
    # Copy config
    with open(evaluation_dir / 'config.yml', 'w') as cfg_file:
        cfg_file.write(cfg.dump())

    # Build evaluation dataset
    evaluation_dataset = build_dataset(cfg, evaluation_dataset_name)

    describe_dataset(evaluation_dataset, logger, save_to=evaluation_dir / 'description.txt')
    
    # Setup Generators
    evaluation_generator = build_generator(cfg, evaluation_dataset)

    # Setup model
    # Load checkpoint model
    input_shapes = evaluation_generator.get_input_shape()
    ensembler = build_ensembler(cfg, input_shapes)
    
    # Evaluation
    start_time = time.time()
    events = []
    uncertainties = []
    for X, event_true in tqdm(evaluation_generator):
        predictions, _, uncertainty = ensembler(X, uncertainty=True)
        events += Event.add_prediction_list(event_true, predictions, ensembler.task)
        uncertainties += [u for u in uncertainty.numpy()]
    evaluation_results = Event.list_to_dataframe(events)
    evaluation_results['uncertainty'] = uncertainties
    evaluation_results = evaluation_results.merge(evaluation_dataset.groupby('event_unique_id').size().rename('cardinality'), on='event_unique_id')
    evaluation_results.to_csv(evaluation_dir / 'results.csv', index=False)
    
    evaluation_time = (time.time() - start_time) / 60.0
    logger.info(f'Evaluation time: {evaluation_time:.3f} [min]')
    
    # Visualizations
    # Regression
    if ensembler.task is Task.REGRESSION:
        # Target Regression
        targets = [t.split('_')[1] for t in cfg.OUTPUT.REGRESSION.TARGETS]
        metrics.targets_regression(evaluation_results, targets)
        # Resolution
        metrics.reconstruction_resolution(evaluation_results, targets, ylim=(0, 2))
        # Theta2 distribution
        metrics.theta2_distribution(evaluation_results, targets)
    # Classification
    if ensembler.task is Task.CLASSIFICATION:
        # Classification Report
        labels = evaluation_generator.output_mapper.classes
        metrics.classification_report(evaluation_results.pred_class_id, evaluation_results.true_class_id, labels=labels)
        metrics.confusion_matrix(evaluation_results.pred_class_id, evaluation_results.true_class_id, labels=labels)


    return evaluation_results, ensembler, evaluation_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a trained neural network model.')
    parser.add_argument('--config-file', required=True, metavar='FILE',
                        help='path to config file')
    parser.add_argument('--use_validation', action='store_true',
                        help='Use "validation set" instead of "test set"')
    parser.add_argument('--dataset_name', default=None, type=str,
                        help='Test dataset name, it used for naming the evaluation folder.')
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
# python evaluate_ensembler.py --config-file /home/asuka/projects/gerumo2/config/regression/umonne/umonne_ensembler.yml --use_validation