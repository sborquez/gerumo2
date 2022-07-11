import logging
import sys; sys.path.append('..')  # noqa
import os
import time
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # noqa

import numpy as np
from gerumo.data.dataset import describe_dataset
from gerumo.data.generators import build_generator
from gerumo.models.base import build_model
from gerumo.utils.structures import Event, Task
from gerumo.utils.engine import (
    setup_cfg, setup_environment, setup_experiment, load_model, build_dataset
)
from gerumo.visualization import metrics
from gerumo.visualization.samples import event_regression


def main(args):
    # Setup
    # Find the config.yml
    if os.path.isdir(args.config_file):
        args.config_file = os.path.join(args.config_file, 'config.yml')
    # Load the configurations
    cfg = setup_cfg(args)
    output_dir, evaluation_dir = setup_experiment(cfg, training=False)
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
    # Build evaluation dataset
    evaluation_dataset = build_dataset(cfg, evaluation_dataset_name)
    describe_dataset(evaluation_dataset, logger, save_to=evaluation_dir / 'description.txt')
    
    # Setup Generators
    evaluation_generator = build_generator(cfg, evaluation_dataset)

    # Setup model
    # Load checkpoint model
    input_shape = evaluation_generator.get_input_shape()
    model = build_model(cfg, input_shape)
    model = load_model(model, evaluation_generator, output_dir, args.epoch)
    
    # Sampling
    n_samples = 15
    random_state = np.random.RandomState(18061996)
    batch_samples = random_state.randint(len(evaluation_generator) - 1, size=n_samples)

    # Evaluation
    start_time = time.time()
    events = []
    uncertainties = []
    for i, (X, event_true) in enumerate(tqdm(evaluation_generator)):
        predictions, y, uncertainty = model(X, uncertainty=True)
        event_predictions = Event.add_prediction_list(event_true, predictions, model.task)
        events += event_predictions
        uncertainties += [u for u in uncertainty.numpy()]
        if (model.task is Task.REGRESSION) and (i in batch_samples):
            j = random_state.randint(len(X))
            targets = cfg.OUTPUT.REGRESSION.TARGETS
            targets_domains = cfg.OUTPUT.REGRESSION.TARGETS_DOMAINS
            # Event true and prediction
            event = event_predictions[j]
            model_output = y[j]
            event_regression(
                event, model_output, model.REGRESSION_OUTPUT_TYPE, targets,
                targets_domains, save_to=str(evaluation_dir)
            )
    evaluation_results = Event.list_to_dataframe(events)
    evaluation_results['uncertainty'] = uncertainties
    evaluation_results['cardinality'] = 1 
    evaluation_results.to_csv(evaluation_dir / 'results.csv', index=False)
    
    evaluation_time = (time.time() - start_time) / 60.0
    logger.info(f'Evaluation time: {evaluation_time:.3f} [min]')
    
    # Visualizations
    # Regression
    if Task[cfg.MODEL.TASK] is Task.REGRESSION:
        # Target Regression
        targets = [t.split('_')[1] for t in cfg.OUTPUT.REGRESSION.TARGETS]
        metrics.targets_regression(evaluation_results, targets, save_to=evaluation_dir)
        # Resolution
        metrics.reconstruction_resolution(evaluation_results, targets, ylim=(0, 2), save_to=evaluation_dir)
        # Theta2 distribution
        metrics.theta2_distribution(evaluation_results, targets, save_to=evaluation_dir)
    # Classification
    if model.task is Task.CLASSIFICATION:
        # Classification Report
        labels = evaluation_generator.output_mapper.classes
        metrics.classification_report(evaluation_results.pred_class_id, evaluation_results.true_class_id, labels=labels, save_to=evaluation_dir)
        metrics.confusion_matrix(evaluation_results.pred_class_id, evaluation_results.true_class_id, labels=labels, save_to=evaluation_dir)

    return evaluation_results, model, evaluation_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a trained neural network model.')
    parser.add_argument('--config-file', required=True, metavar='FILE',
                        help='path to config file')
    parser.add_argument('--epoch', default=-1, type=int,
                        help='Which epoch use to evaluate.')
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
