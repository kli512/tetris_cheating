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

        pts = normalize_pts_xeq1(points)
        pts2 = np.roll(pts, 1, axis=0)
        self.lines = np.array(list(zip(pts, pts2)))

    def match_contour(self, pts, normalize=False):
        if normalize:
            normalized_pts = normalize_pts_xeq1(pts)
        else:
            normalized_pts = np.array(pts)

        return points_to_segs(normalized_pts, self.lines)



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

def match_shape(pts, normalize=False):
    def rotate(points, rots):
        for _ in range(rots):
            points = points @ R.T
        return points

    def match_shape(shape):
        return min(shape.match_contour(rotate(pts, rots), normalize=normalize) for rots in range(4))

    return min(Shapes, key=match_shape).name

# def match_shape(pts, normalize=False):
#     return min(Shapes, key=lambda s: s.match_contour(pts, normalize=normalize)).name

if __name__ == '__main__':
    points = [(0.8620689655172413, 0.0),
              (0.3448275862068966, 0.0),
              (0.3448275862068966, 0.3103448275862069),
              (0.3103448275862069, 0.3448275862068966),
              (0.0, 0.3448275862068966),
              (0.0, 0.6551724137931034),
              (0.6551724137931034, 0.6551724137931034),
              (0.6551724137931034, 0.3448275862068966),
              (0.6896551724137931, 0.3103448275862069),
              (1.0, 0.3103448275862069),
              (1.0, 0.0),
              (0.8620689655172413, 0.0)]
    for shape in Shapes:
        print(f'Shape {shape.name} had error {shape.match_contour(points)}')
