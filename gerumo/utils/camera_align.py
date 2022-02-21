import numpy as np


def compute_alignment(camera_type: str,
                      pixels_positions: np.ndarray) -> np.ndarray:
    if camera_type == 'LSTCam':
        pixels_positions = LST_LSTCam_align(pixels_positions)
    if camera_type == 'CHEC':
        return SST_CHEC_to_indices(pixels_positions)
    else:
        return to_simple_and_shift(pixels_positions)


def LST_LSTCam_align(pixpos: np.ndarray) -> np.ndarray:
    xs, ys = pixpos
    # Distance matrix:
    delta_x = np.array([xs]) - np.array([xs]).T
    delta_y = np.array([ys]) - np.array([ys]).T
    dists = (delta_x**2 + delta_y**2)**0.5
    angles = np.arctan2(delta_y, delta_x)  # Angles from -pi to pi
    # Binary search, find maximum radious where no cell has more
    # than 6 neighbors
    rad1 = 0
    rad2 = np.max(dists)
    for i in range(1000):
        rad = (rad1 + rad2) / 2.0
        neighs = dists < rad  # matrix with true if i,j are neighbors
        np.fill_diagonal(neighs, False)
        max_neighs = np.max(np.sum(neighs, axis=1))
        if max_neighs > 6:
            rad2 = rad
        else:
            rad1 = rad
    #
    rad = rad1
    neighs = dists < rad
    # Get a group of angles on an interval:
    ang_start = 0
    ang_end = np.pi * (6 // 2)
    # Neighbors with angle between those two
    conditions = np.all(
        [neighs, angles >= ang_start, angles < ang_end], axis=0)
    neighbors = np.where(conditions)
    neigh_angles = angles[neighbors]
    # From the angles in this group, pick the median as the main axis
    main_axis_ang = np.median(neigh_angles)
    main_x = np.cos(main_axis_ang)
    main_y = np.sin(main_axis_ang)
    # Apply transformation
    tx = xs * main_x + ys * main_y
    ty = xs * main_y - ys * main_x
    # Now compute the maximum separation between neighboors in the main axis. # noqa
    dx = np.max(delta_x[neighs] * main_x + delta_y[neighs] * main_y)
    # Scale main axis by half of that separation:
    tx = np.round(tx / (dx / 2.0))
    # Now compute the maximum separation between neighboors in the secondary axis. # noqa
    dy = np.max(delta_x[neighs] * main_y - delta_y[neighs] * main_x)
    # Scale secondary axis by that separation:
    ty = np.round(ty / dy)
    return np.stack((tx, ty))


def SST_CHEC_to_indices(pixpos: np.ndarray) -> np.ndarray:
    new_pixpos = pixpos.copy()
    new_pixpos = pixpos.copy()
    new_pixpos -= new_pixpos.min(axis=1, keepdims=True)
    xs, ys = (100 * new_pixpos / new_pixpos.max(axis=1, keepdims=True)).astype(int)
    new_xs = np.zeros_like(xs)
    new_ys = np.zeros_like(ys)
    for i, yl in enumerate(np.sort(np.unique(ys))):
        new_ys[abs(ys - yl) < 2] = i
    for j, xl in enumerate(np.sort(np.unique(xs)[:-1])):
        new_xs[abs(xs - xl) < 2] = j
    new_pixpos = np.vstack((new_xs, new_ys))
    return new_pixpos, new_pixpos


def to_simple_and_shift(pixpos: np.ndarray) -> np.ndarray:
    # get pixels positions
    xs, ys = pixpos
    # indices of x and y pixels position
    i = np.arange(0, len(ys))
    # row values of the telescope
    y_levels = np.sort(np.unique(ys))
    # image dimension
    nrows = len(y_levels)
    # new translated pixel positions
    new_x_l = np.copy(xs)  # new pixels x positions left shift
    new_x_r = np.copy(xs)  # new pixels x positions right shift
    new_y = np.copy(ys)
    # shift odd rows
    dx = 0
    for level, y_value in enumerate(y_levels):
        indices = i[ys == y_value]
        if dx == 0:
            dx = np.diff(np.sort(xs[indices])).min() / 2
        if level % 2 != 0:
            new_x_l[indices] -= dx
            new_x_r[indices] += dx
    # round values
    new_x_l = np.round(new_x_l, 3)
    new_x_r = np.round(new_x_r, 3)
    # max indices of image output
    max_col_l = len(np.unique(new_x_l)) - 1
    max_col_r = len(np.unique(new_x_r)) - 1
    max_row = nrows - 1
    # apply linear transfomation
    new_x_l = ((max_col_l/(new_x_l.max() - new_x_l.min())) * (new_x_l - new_x_l.min()))  # noqa
    new_x_l = np.round(new_x_l).astype(int)
    new_x_r = ((max_col_r/(new_x_r.max() - new_x_r.min())) * (new_x_r - new_x_r.min()))  # noqa
    new_x_r = np.round(new_x_r).astype(int)
    new_y = ((max_row/(new_y.max() - new_y.min())) * (new_y - new_y.min()))  # noqa
    new_y = np.round(new_y).astype(int)
    # prepare output
    simple = np.vstack((new_x_r, new_y))
    simple_shift = np.vstack((new_x_l, new_y))
    return simple, simple_shift
