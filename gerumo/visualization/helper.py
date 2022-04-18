import matplotlib


import matplotlib.pyplot as plt
from os.path import join


"""
Helper function
================
"""


def show_or_save(figure, save_to, title):
    if save_to is not None:
        figure.savefig(join(save_to, title))
        plt.close(figure)
    else:
        plt.show()


def label_formater(target, use_degrees=True, only_units=False):
    if target.startswith('true_') or target.startswith('pred_'):
        target = target[5:] 
    units = {
        'az': '[deg]' if use_degrees else '[rad]',
        'alt': '[deg]' if use_degrees else '[rad]',
        'mc_energy': '[TeV]',
        'log10_mc_energy': '$log_{10}$ [TeV]'
    }
    if only_units:
        return units[target]
    else:
        return f'{target} {units[target]}'
