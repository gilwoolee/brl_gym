import numpy as np

def convert_3D_to_2D(xy, maze_type=4):
    """
    Convert point_mass.py's x,y to this
    """
    xy = xy.copy()
    if maze_type == 4:
        xy = ((xy + 1.5) * 100).astype(np.int)
    else:
        if len(xy.shape) == 1:
            xy[1] *= -1
        else:
            xy[:, 1] *= -1
        xy = ((xy + 1.5) * 101. / 3.0).astype(np.int)
        if len(xy.shape) == 1:
            xy = np.array([xy[1], xy[0]])
        else:
            xy = np.concatenate([xy[:, [1]], xy[:, [0]]], axis=1)
    return xy

def convert_2D_to_3D(xy, maze_type=4):
    xy = xy.copy()
    if maze_type == 4:
        xy = (xy / 100.) - 1.5
    else:
        xy = (xy / 101.) * 3.0 - 1.5

        if len(xy.shape) == 1:
            xy = np.array([xy[1], xy[0]])
        else:
            xy = np.concatenate([xy[:, [1]], xy[:, [0]]], axis=1)

        if len(xy.shape) == 1:
            xy[1] *= -1
        else:
            xy[:, 1] *= -1

    return xy

# xy = convert_3D_to_2D(np.array([0.65, 1.325]), maze_type=10)
# print([0.65, 1.325], xy)

# xy = convert_2D_to_3D(xy, maze_type=10)
# print(xy)