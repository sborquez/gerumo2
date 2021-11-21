"""
Exploration Visualizations
======================

Generate plot for display dataset, images and targets.

Here you can find dataset exploration and samples visualizations.
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np



"""
Input Samples
================
"""

"""
Target Samples
================
"""

"""
Dataset Visualizations
================
"""

def plot_array(hdf5_file, layout="configuration/instrument/subarray/layout"):
    plt.figure(figsize=(8, 8))
    plt.title("Array info")
    markers = ['x', '+', 'v',  '^', ',','<', '>', 's',',', 'd']
    telescopes_groups = {}
    for row in hdf5_file.root[layout]:
        telescope_type = row["type"].decode()+"-"+row["camera_type"].decode()
        if telescope_type not in telescopes_groups:
            telescopes_groups[telescope_type] = {"x": [], "y":[], "marker": markers.pop()}
        telescopes_groups[telescope_type]["x"].append(row["pos_x"])
        telescopes_groups[telescope_type]["y"].append(row["pos_y"])
    for k,v in telescopes_groups.items():
        plt.scatter(v["x"], v["y"], label=k, marker=v["marker"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def plot_telescope_geometry(tel_type, pixel_positions, num_pixels=None):
    shape = pixel_positions.shape
    if num_pixels is None:
        num_pixels = shape[-1]
    colors = cm.rainbow(np.linspace(0, 1, num_pixels))
    if shape[0] == 2:
        plt.figure(figsize=(8,8))
        plt.title(f"{tel_type}\n pixels: {num_pixels}")
        plt.scatter(pixel_positions[0], pixel_positions[1], color=colors)
    elif shape[0] == 3:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        plt.suptitle(f"{tel_type}\n pixels: {num_pixels}")        
        ax1.scatter(pixel_positions[0], pixel_positions[2], color=colors)
        ax2.scatter(pixel_positions[1], pixel_positions[2], color=colors)
    plt.show()


def plot_observation_scatter(charge, peakpos, pixel_positions, telescope_type=None, event_unique_id=None):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    ax1.set_title("Charge")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax1.scatter(pixel_positions[0], pixel_positions[1], c=charge)
    plt.colorbar(im, cax=cax)

    ax2.set_title("Peak Pos")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax2.scatter(pixel_positions[0], pixel_positions[1], c=peakpos)
    plt.colorbar(im, cax=cax)

    if event_unique_id is None:
        title = f"{'' if telescope_type is None else telescope_type}"
    else:
        title = f"{'' if telescope_type is None else telescope_type}\nevent: {event_unique_id}"

    plt.suptitle(title)
    plt.show()
