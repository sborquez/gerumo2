"""
Metrics Visualizations
======================

Generate plot for different metrics of models.

Here you can find training metrics, single model evaluation
and models comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import ctaplot
import astropy.units as u

from .helper import *

"""
Training Metrics
================
"""


def training_history(history, training_time, model_name,
                     ylog=False, save_to=None):
    """
    Display training loss and validation loss vs epochs.
    """
    fig = plt.figure(figsize=(12, 6))
    epochs = len(history.history['loss'])  # fix: early stop
    epochs = [i for i in range(1, epochs + 1)]
    plt.plot(epochs, history.history['loss'], '--', label='Train')
    plt.plot(epochs, history.history['val_loss'], '--', label='Validation')

    # Style
    title = f'Model {model_name} Training Loss\n '
    title += f'Training time {training_time:.2f} [min]'
    plt.title(title)
    if ylog:
        plt.ylabel('log Loss')
        plt.yscale('log')
    else:
        plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.xticks(rotation=-90)
    plt.grid(True)

    # Show or Save
    if ylog:
        show_or_save(fig, save_to, title=f'{model_name} - Training Loss.png')
    else:
        show_or_save(fig, save_to, title=f'{model_name} - Training Log Loss.png')
    

"""
Evaluation Metrics
================
"""


def targets_regression(evaluation_results, targets, save_to=None):
    """
    Display regression metrics for a model's predictions.
    
    Example:
    ========
    ```
    results = {
      'pred_alt': [1, ...],
      'true_alt': [1, ...],
      'pred_az':  [1, ...],
      'true_az':  [1, ...],
    }
    targets = ['alt', 'az']
    evaluation_results = pd.DataFrame(results)
    plot_model_validation_regressions(evaluation_results, targets)
    ```
    """
    n_targets = len(targets)
    # Create Figure and axis
    fig, axs = plt.subplots(n_targets, 3, figsize=(19, n_targets * 6))
    
    # Style
    fig.suptitle('Targets Regression')

    # For each target, generate two plots
    for i, target in enumerate(targets):
        # Target and prediction values
        prediction_points = evaluation_results[f'pred_{target}']
        targets_points = evaluation_results[f'true_{target}']
        score = r2_score(prediction_points, targets_points)
        # Show regression
        ax_r = axs[i][0] if n_targets > 1 else axs[0]
        add_regression_identity(prediction_points, targets_points, score, target, flip=True, axis=ax_r)
        # Show error
        ax_e = axs[i][1] if n_targets > 1 else axs[1]
        add_residual_error(prediction_points, targets_points, target, axis=ax_e)
        # Show error distribution
        ax_d = axs[i][2] if n_targets > 1 else axs[2]
        add_residual_error_distribution(prediction_points, targets_points, target, vertical=True, axis=ax_d)
    
    show_or_save(fig, save_to, 'Targets Regression.png')


def add_regression_identity(prediction_points, targets_points, score, target, flip=False, axis=None):
    """
    Show a comparation between true values and predicted values, it uses a scatter plot
    for a small set or a hexbin plot for a set bigger than 500 samples.

    A nice fit means that points are distributed close to the identity diagonal
    """
    if flip:
        ylabel = 'Predicted Values'
        y = prediction_points.values
        xlabel = 'True Values'
        x = targets_points.values
    else:
        ylabel = 'True Values'
        y = targets_points.values
        xlabel = 'Predicted Values'
        x = prediction_points.values

    # Create new figure
    if axis is None:
        plt.figure(figsize=(6, 6))
        axis = plt.gca()

    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    if len(targets_points) < 500:
        axis.scatter(x=x, y=y, alpha=0.6)
        # Add identity line
        axis.plot([vmin, vmax], [vmin, vmax], 'r--', label='identity', linewidth=3)
    else:
        x = np.append(x, vmin)
        x = np.append(x, vmax)
        y = np.append(y, vmin)
        y = np.append(y, vmax)
        axis.hexbin(x, y, gridsize=(41, 41), cmap='jet')
        # Add identity line
        axis.plot([vmin, vmax], [vmin, vmax], 'w--', label='identity', linewidth=3)

    # Style
    title = label_formater(target)
    axis.set_title(f'Regression on {title}')
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.grid(True)
    axis.set_aspect('equal')

    # Empty plot for add legend score
    axis.plot([], [], ' ', label=f'$R^2$ score = {score:.4f}')
    axis.legend()

    return axis


def add_residual_error(prediction_points, targets_points, target, axis=None):
    """
    Show the distribution of the residual error along the predicted points.

    The residual error is calculated as the diference of targets_points and
    prediction_points:

    residual_error = targets_points - prediction_points
    """

    # Create new figure
    if axis is None:
        plt.figure(figsize=(6, 6))
        axis = plt.gca()
    
    # Residual Error
    residual_error = targets_points - prediction_points
    x_vmin = prediction_points.min()
    x_vmax = prediction_points.max()
    y_vmin = residual_error.min()
    y_vmax = residual_error.max()
    y_lim = 1.05 * max(abs(y_vmin), abs(y_vmax))

    # Plot
    if len(residual_error) < 500:
        axis.scatter(x=prediction_points, y=residual_error, alpha=0.6)
    else:
        x = prediction_points.values
        y = residual_error.values
        x = np.append(x, x_vmin)
        x = np.append(x, x_vmax)
        y = np.append(y, -1 * y_lim)
        y = np.append(y, y_lim)
        axis.hexbin(x, y, gridsize=(41, 41), cmap='jet', zorder=0)
    axis.plot([x_vmin, x_vmax], [0, 0], 'r--')

    # Style
    title = label_formater(target)
    axis.set_title(f'Residual Error on {title}')
    axis.set_xlabel('Predicted Values')
    axis.set_ylabel('Residual Error')
    axis.set_ylim([-1 * y_lim, y_lim])
    axis.grid(True)
    axis.set_aspect('auto')
    return axis


def add_residual_error_distribution(prediction_points, targets_points, target, vertical=False, axis=None):
    """
    Show the distribution of the residual error, caculate its mean and std.

    residual_error = targets_points - prediction_points
    """
    # Create new figure
    if axis is None:
        plt.figure(figsize=(6, 6))
        axis = plt.gca()
    
    # Residual Error
    residual_error = targets_points - prediction_points
    vmin = residual_error.min()
    vmax = residual_error.max()
    lim = 1.05 * max(abs(vmin), abs(vmax))

    # Normalized
    weights = np.ones_like(residual_error) / len(residual_error)

    # Plot
    unit = label_formater(target, only_units=True)
    legend = f'mean: {residual_error.mean():.4f} {unit}\nstd: {residual_error.std():.4f} {unit}'
    if vertical:
        axis.hist(residual_error, weights=weights, bins=40, range=(-1 * lim, lim),
                  orientation='horizontal', label=legend)
    else:
        axis.hist(residual_error, weights=weights, bins=40, range=(-1 * lim, lim),
                  label=legend)

    # Style
    title = label_formater(target)
    axis.set_title(f'Residual Error Distribution on {title}')
    if vertical:
        axis.set_ylabel('Residual Error')
        axis.set_ylim([-1 * lim, lim])
    else:
        axis.set_xlabel('Residual Error')
        axis.set_xlim([-1 * lim, lim])
    axis.set_aspect('auto')
    axis.legend()
    axis.set_aspect('auto')
    axis.legend()
    return axis


"""
CTA Metrics
===========
"""


def reconstruction_resolution(evaluation_results, targets, use_pred_energy=False, save_to=None, **kwargs):
    # Angular Resolution
    if set(['az', 'alt']).issubset(targets):
        if use_pred_energy:
            assert 'pred_energy' in evaluation_results.columns
            energy_col = 'pred_energy'
        else:
            energy_col = 'true_energy'
        figure = plt.figure(figsize=(6, 6))
        
        add_angular_resolution(
            pred_alt=evaluation_results['pred_alt'].values * u.deg,
            pred_az=evaluation_results['pred_az'].values * u.deg,
            true_alt=evaluation_results['true_alt'].values * u.deg,
            true_az=evaluation_results['true_az'].values * u.deg,
            energy=evaluation_results[energy_col].values * u.TeV,
            ax=plt.gca(),
            **kwargs
        )
        show_or_save(figure, save_to, 'Angular Resolution.png')
    # Energy Resolution
    if 'energy' in targets:
        figure = plt.figure(figsize=(6, 6))
        add_angular_resolution(
            pred_energy=evaluation_results['pred_energy'].values * u.TeV,
            true_energy=evaluation_results['true_energy'].values * u.TeV,
            ax=plt.gca(),
            **kwargs
        )
        show_or_save(figure, save_to, 'Energy Resolution.png')
    

def add_angular_resolution(pred_alt, pred_az, true_alt, true_az, energy,
                           percentile=68.27, confidence_level=0.95, bias_correction=False,
                           label='this method', include_requirement=[],
                           xlim=None, ylim=None, fmt=None, ax=None, **kwargs):
    """
    Show absolute angular error for a model's predictions.
    """
    # Create new figure
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    # Style
    fmt = fmt or 'o'
    # Ctaplot - Angular resolution
    ax = ctaplot.plot_angular_resolution_per_energy(
        true_alt, pred_alt, true_az, pred_az, energy,
        percentile, confidence_level, bias_correction, ax,
        fmt=fmt, label=label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.xaxis.grid(False, which='minor')
    try:
        for include in include_requirement:
            ax = ctaplot.plot_energy_resolution_cta_requirement(include, ax)
    except ValueError:
        print('Unable to display cta requirements.')
    return ax


def add_energy_resolution(pred_energy, true_energy,
                          percentile=68.27, confidence_level=0.95, bias_correction=False,
                          label='this method', include_requirement=[],
                          xlim=None, ylim=None, fmt=None, ax=None, **kwargs):
    """
    Show the energy resolution for a model's predictions.
    """
    
    # Create new figure
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    fmt = fmt or 'o'
    ax = ctaplot.plot_energy_resolution(
        true_energy, pred_energy, percentile=percentile,
        confidence_level=confidence_level, bias_correction=bias_correction,
        fmt=fmt, label=label, ax=ax
    )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.xaxis.grid(False, which='minor')
    try:
        for include in include_requirement:
            ax = ctaplot.plot_energy_resolution_cta_requirement(include, ax)
    except ValueError:
        print('Unable to display cta requirements.')
    return ax


def theta2_distribution(evaluation_results, targets, save_to=None, **kwargs):
    if set(['az', 'alt']).issubset(targets):
        figure = plt.figure(figsize=(6, 6))
        
        add_absolute_error_angular(
            pred_alt=evaluation_results['pred_alt'].values * u.deg,
            pred_az=evaluation_results['pred_az'].values * u.deg,
            true_alt=evaluation_results['true_alt'].values * u.deg,
            true_az=evaluation_results['true_az'].values * u.deg,
            ax=plt.gca(),
            **kwargs
        )
        show_or_save(figure, save_to, 'Theta 2 Distribution.png')


def add_absolute_error_angular(pred_alt, pred_az, true_alt, true_az, bias_correction=False, ax=None):
    """
    Show the absolute error distribution of a method.
    """
    # Create new figure
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    
    bins = np.linspace(0.001, 2, 50)
    ax = ctaplot.plot_theta2(true_alt, pred_alt, true_az, pred_az, bias_correction, ax, bins=bins)
    return ax
