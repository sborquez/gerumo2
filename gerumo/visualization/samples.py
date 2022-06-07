from typing import List, Optional

import numpy as np
import tensorflow as tf
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import kde

from .helper import *
from ..utils.structures import Event, OutputType


def event_regression(event: Event, model_output: tf.Tensor, output_type: OutputType, 
                     targets: List[str], targets_domains: List[List[int]], save_to: Optional[str] = None) -> None:
    """
    Display event prediction, the probability and the predicted point.
    """

    # Create new Figure
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    
    # Style
    title = f"Prediction for event {event.event_unique_id}"
    plt.title(title)

    # Point estimator
    if output_type is OutputType.POINT:
        if len(targets) == 1:
            raise NotImplementedError
        elif len(targets) == 2:
            ax = add_2d_point_prediction(event, targets, axis=ax)
        elif len(targets) == 3:
            raise NotImplementedError

    # Monte-Carlo samples estimator
    elif output_type is OutputType.SAMPLES:
        # Show prediction according to targets dim
        if len(targets) == 1:
            raise NotImplementedError
        elif len(targets) == 2:
            ax = add_kde_2d(model_output, targets, targets_domains, axis=ax)
            ax = add_2d_point_prediction(event, targets, axis=ax)
        elif len(targets) == 3:
            raise NotImplementedError

    # Probability mass function estimator
    elif output_type is OutputType.PMF:
        # Show prediction according to targets dim
        if len(targets) == 1:
            raise NotImplementedError
        elif len(targets) == 2:
            ax = add_pmf_2d(model_output, targets, targets_domains, axis=ax)
            ax = add_2d_point_prediction(event, targets, axis=ax)
        elif len(targets) == 3:
            raise NotImplementedError

    else:
        raise ValueError('Output type not supported', output_type)
    
    
    # Save or Show
    show_or_save(fig, save_to, title=f'{title}.png')

"""
Event prediction
"""
def add_1d_point_prediction(event: Event, targets: List[str], axis: plt.Axes):
    assert len(targets) == 1
    raise NotImplementedError
    # Create new figure
    if axis is None:
        plt.figure(figsize=(16,8))
        axis = plt.gca()
    # Alt always vertical
    priority = ('true_alt', 'true_az', 'true_energy', 'true_log_energy', 'true_particle_type')
    sorted_targets = [p for p in priority if p in targets]
    
    # Get data points
    data = event.get_fields()
    # Add target point
    x_target = data[sorted_targets[1]]
    # axis.scatter(
    #     x=[x_target],
    #     c='black',marker='o', 
    #     label=f'target=({x_target:.4f})', alpha=0.9
    # )
    # Add predicted point
    sorted_predictions = [target.replace('true', 'pred') for target in sorted_targets]
    if not ((sorted_predictions[0] in data) and (sorted_predictions[1] in data)):
        return axis
    #x_prediction = data[sorted_predictions[0]]
    # axis.scatter(
    #     x=[x_prediction],
    #     c='white', marker='*', 
    #     label=f'prediction=({x_prediction:.4f})', alpha=0.9
    # )
    return axis


def add_2d_point_prediction(event: Event, targets: List[str], axis: plt.Axes):
    assert len(targets) == 2
    # Create new figure
    if axis is None:
        plt.figure(figsize=(16,8))
        axis = plt.gca()
    # Alt always vertical
    priority = ('true_alt', 'true_az', 'true_energy', 'true_log_energy', 'true_particle_type')
    sorted_targets = [p for p in priority if p in targets]
    
    # Get data points
    data = event.get_fields()
    # Add target point
    x_target = data[sorted_targets[1]]
    y_target = data[sorted_targets[0]]
    axis.scatter(
        x=[x_target],
        y=[y_target],
        c='black',marker='o', 
        label=f'target=({x_target:.4f}, {y_target:.4f})', alpha=0.9
    )
    # Add predicted point
    sorted_predictions = [target.replace('true', 'pred') for target in sorted_targets]
    if not ((sorted_predictions[0] in data) and (sorted_predictions[1] in data)):
        return axis
    x_prediction = data[sorted_predictions[1]]
    y_prediction = data[sorted_predictions[0]]
    axis.scatter(
        x=[x_prediction],
        y=[y_prediction],
        c='white', marker='*', 
        label=f'prediction=({x_prediction:.4f}, {y_prediction:.4f})', alpha=0.9
    )
    plt.legend()
    return axis

"""
PMF predictions
--------------
"""

def add_pmf_1d(prediction, targets, targets_domains, axis=None):
    """
    Show predicted pmf in a 1d target domain.
    """
    raise NotImplementedError
    assert len(targets) == 1
    if isinstance(prediction, tf.Tensor):
        prediction = prediction.numpy()
    # Create new figure
    if axis is None:
        plt.figure(figsize=(16,8))
        axis = plt.gca()
    axis.set_xlabel(label_formater(targets[target_order[0]]))
    axis.grid(False)
    return axis

def add_pmf_2d(prediction, targets, targets_domains, axis=None):
    """
    Show predicted pmf in a 2d target domain.
    """
    assert len(targets) == 2
    if isinstance(prediction, tf.Tensor):
        prediction = prediction.numpy()
    # Create new figure
    if axis is None:
        plt.figure(figsize=(16,8))
        axis = plt.gca()

    # Alt always vertical
    priority = ('true_alt', 'true_az', 'true_energy', 'true_log_energy', 'true_particle_type')
    target_order = [targets.index(p) for p in priority if p in targets]
    
    prediction = prediction.transpose(*target_order)
    extend = [(targets_domains[t][0], targets_domains[t][1]) for t in target_order[::-1]]
    extend = tuple([item for sublist in extend for item in sublist])

    # Draw probability map
    ## Probability map in Log scale
    epsilon = 2e-10
    vmin = prediction.min()
    if vmin <= 0:
      vmin = epsilon
      prediction += epsilon
    vmax = prediction.max()
    im = axis.imshow(prediction, origin="lower", cmap="jet", extent=extend, 
                     aspect=3,  norm=LogNorm(vmin=vmin, vmax=vmax))
    ## Add color bar
    plt.colorbar(im, ax=axis, extend='max')

    # Style
    axis.set_ylabel(label_formater(targets[target_order[0]]))
    axis.set_xlabel(label_formater(targets[target_order[1]]))
    axis.grid(False)
    return axis


"""
Samples predictions
-----------------------
"""

def add_kde_2d(prediction, targets, targets_domains, axis=None):
    """
    Show predicted samples in a 2d target domain.
    """
    assert len(targets) == 2
    if isinstance(prediction, tf.Tensor):
        prediction = prediction.numpy()
    assert prediction.ndim == 2, 'dimensions should be (samples, targets)'
    # Create new figure
    if axis is None:
        plt.figure(figsize=(16,8))
        axis = plt.gca()

    # Alt always vertical
    nbins = 100
    priority = ('true_alt', 'true_az', 'true_energy', 'true_log_energy', 'true_particle_type')
    target_order = [targets.index(p) for p in priority if p in targets]
    # print(target_order)
    y = prediction[:, target_order[0]]
    x = prediction[:, target_order[1]]
    k = kde.gaussian_kde([x,y])
    extend = [(targets_domains[t][0], targets_domains[t][1]) for t in target_order[::-1]]
    extend = tuple([item for sublist in extend for item in sublist])
    xx, yy = np.mgrid[extend[0]:extend[1]:nbins*1j, extend[2]:extend[3]:nbins*1j]
    zi = k(np.vstack([xx.flatten(), yy.flatten()]))
    
    # Draw probability map
    ## Probability map in Log scale
    # Change color palette
    axis.set_aspect((extend[1] - extend[0])/(extend[3]-extend[2]))
    epsilon = 2e-10
    vmin = zi.min()
    if vmin <= 0:
      vmin = epsilon
      zi += epsilon
    vmax = prediction.max()
    im = plt.pcolormesh(xx, yy, zi.reshape(xx.shape), shading='auto',
                        cmap=plt.cm.jet, norm=LogNorm(vmin=vmin, vmax=vmax))
    ## Add color bar
    plt.colorbar(im, ax=axis, extend='max')

    # Style
    axis.set_ylabel(label_formater(targets[target_order[0]]))
    axis.set_xlabel(label_formater(targets[target_order[1]]))
    axis.grid(False)
    return axis

