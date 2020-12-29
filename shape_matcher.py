import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

'''
just switch back to the old affine transformation/area overlap solution...
'''



def points_to_segs(pts, segs):
    diffs = segs[:, 1, :] - segs[:, 0, :]

    mags = np.sum(diffs ** 2, axis=-1)
    dists = pts[None, :, :] - segs[:, None, 0, :]

    ts = np.clip(
        np.sum(dists * diffs[:, None, :], axis=-1) / mags[:, None], 0, 1)
    ds = ts[:, :, None] @ diffs[:, None, :]

    closest_pts = segs[:, 0, :] + np.swapaxes(ds, 0, 1)

    pt_to_line_distances = np.sqrt(
        np.sum((closest_pts - pts[:, None, :]) ** 2, axis=-1))

    return np.sum(np.min(pt_to_line_distances, axis=-1))


'''

pts = np.array([[0. , 0.5],
       [2. , 0. ],
       [1. , 1. ],
       [1.5, 10]])

segs = np.array([
        [[0, 1],
        [0, 0]],

       [[3, 0],
        [0, 0]],

       [[3, 3],
        [1, 1]]])
'''

def normalize_shape(shape):
    scaled = affinity.scale(shape, 1 / shape.area ** 0.5, 1 / shape.area ** 0.5)
    centered = affinity.translate(scaled, -scaled.centroid.x, -scaled.centroid.y)
    return centered


def normalize_pts_xeq1(pts):
    pts_arr = np.array(pts)
    mean_pt, max_pt = np.mean(pts_arr, axis=0), np.max(pts_arr, axis=0)
    return (pts_arr - mean_pt) / (max_pt[0] - mean_pt[0])


class Shape:
    def __init__(self, name, points):
        self.name = name
        shape = Polygon(points)
        if shape.area != 4:
            raise ValueError(f'Shape {name} invalid points - area != 4')

        self.shape = normalize_shape(shape)

    def match_shape(self, shape, normalize=False):
        if normalize:
            shape = normalize_shape(shape)
        return shape.intersection(self.shape).area


S = Shape(
    'S',
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 2],
        [3, 2],
        [3, 1],
        [2, 1],
        [2, 0]
    ]
)

Z = Shape(
    'Z',
    [
        [3, 0],
        [3, 1],
        [2, 1],
        [2, 2],
        [0, 2],
        [0, 1],
        [1, 1],
        [1, 0]
    ]
)

T = Shape(
    'T',
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 2],
        [2, 2],
        [2, 1],
        [3, 1],
        [3, 0]
    ]
)

O = Shape(
    'O',
    [
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
    ]
)

I = Shape(
    'I',
    [
        [0, 0],
        [4, 0],
        [4, 1],
        [0, 1],
    ]
)

L = Shape(
    'L',
    [
        [0, 0],
        [3, 0],
        [3, 2],
        [2, 2],
        [2, 1],
        [0, 1],
    ]
)

J = Shape(
    'J',
    [
        [0, 0],
        [3, 0],
        [3, 1],
        [1, 1],
        [1, 2],
        [0, 2],
    ]
)

Shapes = [S, Z, T, O, I, L, J]

R = np.array([[0, -1], [1, 0]])

def match_shape(input_shape):
    normalized_input_shape = normalize_shape(input_shape)

    def match_rots(shape):
        return max(shape.match_shape(affinity.rotate(normalized_input_shape, rots, origin='centroid')) for rots in range(0, 360, 90))

    return max(Shapes, key=match_rots).name

import matplotlib.pyplot as plt
from descartes import PolygonPatch

def plot_shape(shape):
    ppoints = list(zip(*shape.exterior.coords.xy))
    fig, ax = plt.subplots()
    ax.add_patch(PolygonPatch(shape))
    ax.scatter(*zip(*ppoints))
    plt.show()
