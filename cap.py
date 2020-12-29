from mss import mss
from PIL import Image
import win32gui
import numpy as np
import cv2
from sklearn.cluster import *
import shape_matcher
import tetris_pc_finder
import alphashape
import time
from timeit import default_timer as timer


FAST_BOARD_DETECTION = True
ACTIVE_PIXEL_THRESH = 50
ACTIVE_BOARD_ROWS = 4

BOARD_MASK = np.ones((20, 10))
BOARD_MASK[:20 - ACTIVE_BOARD_ROWS] = 0

def get_monitor(window_name):
    toplist, winlist = [], []
    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
    win32gui.EnumWindows(enum_cb, toplist)

    hwnd = next((hwnd, title) for hwnd, title in winlist if window_name in title.lower())[0]

    bbox = win32gui.GetWindowRect(hwnd)

    return {
        'top': bbox[1],
        'left': bbox[0],
        'width': bbox[2] - bbox[0],
        'height': bbox[3] - bbox[1],
    }

def crop(img, boundaries):
    return img[boundaries[1]:boundaries[3], boundaries[0]:boundaries[2]]

def find_shapes(img, n, pixel_thresh=ACTIVE_PIXEL_THRESH):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    active_pixels = np.argwhere(np.flip(img_gray.T, axis=0) > pixel_thresh)

    # TODO if DBSCAN then may want to resample to lower resolution
    # and resolution of game may affect this later because of eps/min_samples
    try:
        clusters = DBSCAN(eps=5, min_samples=40, metric='manhattan').fit(active_pixels)
    except ValueError:
        raise ValueError('Found 0 clouds in the image') from None

    labels = np.unique(clusters.labels_)

    clouds = [active_pixels[np.argwhere(clusters.labels_ == lbl).flatten()] for lbl in labels]
    if len(clouds) < n:
        raise ValueError(f'Unable to find {n} clouds in the image')
    elif len(clouds) > n:
        print(f'WARNING: more than {n} clouds found in image')

    clouds = sorted(clouds, key=len, reverse=True)[:n]
    clouds = sorted(clouds, key=lambda c: c.min(axis=0)[1])
    # shapes = [alphashape.alphashape(pt_cloud, 1) for pt_cloud in clouds]
    shapes = [alphashape.alphashape(pt_cloud, 1) for pt_cloud in clouds]

    return shapes

    contours = [np.array(list(zip(*shape.exterior.coords.xy))) for shape in shapes]

    contour_means = [contour.mean(axis=0) for contour in contours]
    contour_maxs = [contour.max(axis=0) for contour in contours]

    normalized_contours = [(contour - contour_mean) / (contour_max[0] - contour_mean[0]) for contour, contour_mean, contour_max in zip(contours, contour_means, contour_maxs)]

    # cloud_means = [cloud.mean(axis=0) for cloud in clouds]
    # cloud_maxs = [cloud[:, 0].max() for cloud in clouds]

    # normalized_pts = [(cloud - cloud_mean) / (cloud_max - cloud_mean[0]) for cloud, cloud_mean, cloud_max in zip(clouds, cloud_means, cloud_maxs)]

    return normalized_contours

    # shapes = [alphashape.alphashape(pt_cloud, ALPHA).simplify(0.01) for pt_cloud in normalized_pts]
    # return shapes


# OLD CONTOUR FINDING CODE
# CV2 FIND_CONTOURS SUCKS

# can be made much more efficient... and better written too
# def find_contours(img, n, thresh=1):
#     im_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
#     im_gray = np.uint8((im_gray > 50) * 255)

#     def generate_contours(threshhold):
#         print(f'Using threshold {threshhold}')
#         _, thresh_img = cv2.threshold(im_gray, threshhold, 255, cv2.THRESH_BINARY)
#         return cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

#     contours = generate_contours(thresh)

#     while len(contours) > n:
#         thresh += 1
#         contours = generate_contours(thresh)

#     while len(contours) < n:
#         thresh -= 1
#         contours = generate_contours(thresh)

#     if len(contours) == n:
#         return contours

#     raise ValueError(f'No {n} contour(s) could be found')

def draw_contours(img, contours, color=(255, 255, 255), thickness=3):
    res = img.copy()
    cv2.drawContours(res, contours, -1, color, thickness)

    return res

monitor = get_monitor('tetr.io')

HOLD_BOUNDARIES = (615, 220, 790, 325)
BOARD_BOUNDARIES = (796, 191, 1140, 885)
SPAWNED_BOUNDARIES = (796, 0, 1140, 191)
NEXT_BOUNDARIES = (1164, 220, 1341, 740)

HOLD_SIZE = (175, 105)

if FAST_BOARD_DETECTION:
    BOARD_SIZE = (10, 20)
else:
    BOARD_SIZE = ()
    raise NotImplementedError('Fast board detection only supported')

SPAWNED_SIZE = (344, 191)

# NEXT_SIZE = (100, 294)
NEXT_SIZE = (50, 147)
from collections import deque

def find_pc(board, cur_piece, hold_piece, next_queue, hold_available):
    # print(cur_piece)
    # print(hold_piece)
    # print(next_queue)
    # print(hold_available)

    if np.sum(board == 0) > 20:
        raise ValueError('too much empty stuff')

    next_queue = deque(tetris_pc_finder.Piece(l) for l in next_queue)
    cur_piece = tetris_pc_finder.Piece(cur_piece)
    hold_piece = tetris_pc_finder.Piece(hold_piece)

    state = tetris_pc_finder.TetrisPCState(board, cur_piece, hold_piece, next_queue, hold_available, [])

    res = tetris_pc_finder.dfs(state)
    tetris_pc_finder.plot_solution(state.board, res.path)

def analyze_board(img, prev_state):
    hold_img = cv2.resize(crop(img, HOLD_BOUNDARIES), HOLD_SIZE)
    board_img = cv2.resize(crop(img, BOARD_BOUNDARIES), BOARD_SIZE)
    # spawned_img = cv2.resize(crop(im, SPAWNED_BOUNDARIES), SPAWNED_SIZE)
    next_img = cv2.resize(crop(img, NEXT_BOUNDARIES), NEXT_SIZE)

    next_shapes = find_shapes(next_img, 5)
    next_pieces = tuple(shape_matcher.match_shape(shape) for shape in next_shapes)

    try:
        hold_shape = find_shapes(hold_img, 1)[0]
    except ValueError:
        hold_piece = None
    else:
        hold_piece = shape_matcher.match_shape(hold_shape)

    if prev_state is None or (hold_piece, next_pieces) == prev_state:
        return (hold_piece, next_pieces)

    print('\nFound new board!')
    if hold_piece != prev_state[0]:
        hold_available = False
        if prev_state[0] is None:
            cur_piece = prev_state[1][0]
        else:
            cur_piece = prev_state[0]
    else:
        hold_available = True
        cur_piece = prev_state[1][0]

    if FAST_BOARD_DETECTION:
        board = np.logical_and(cv2.cvtColor(board_img, cv2.COLOR_BGRA2GRAY) > 50, BOARD_MASK)
    else:
        raise NotImplementedError('Fast board detection only supported')

    print('-' * 40)
    print(f'Current piece: {cur_piece}')
    print(f'Hold piece: {hold_piece}')
    print(f'Hold available: {hold_available}')
    print(f'Next queue: {next_pieces}')

    try:
        find_pc(board[16:], cur_piece, hold_piece, next_pieces, hold_available)
    except ValueError as e:
        print(e)

    return (hold_piece, next_pieces)

def main():
    piece_state = None
    # hold_available = True
    # cur_piece = None

    with mss() as sct:
        # while True:
        for _ in range(30):
            # time.sleep(0.1)
            input()
            # print('Grabbing screen...')
            sct_img = sct.grab(monitor)
            # img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

            # hold = img.crop(HOLD_BOUNDARIES)
            # board = img.crop(BOARD_BOUNDARIES)
            # spawned = img.crop(SPAWNED_BOUNDARIES)
            # next_pieces = img.crop(NEXT_BOUNDARIES)
            # print(f'Piece state: {piece_state}')
            piece_state = analyze_board(np.array(sct_img), piece_state)

            # im = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2GRAY)





if __name__ == '__main__':
    main()


# n=5

# img_gray = cv2.cvtColor(next_pieces, cv2.COLOR_BGRA2GRAY)
# pts = np.argwhere(img_gray > 50)

# # clusters = DBSCAN(eps=5, min_samples=40, metric='manhattan').fit(pts)
# clusters = AgglomerativeClustering(n_clusters=n).fit(pts)

# all_pts = [pts[np.argwhere(clusters.labels_ == lbl).flatten()] for lbl in range(5)]

# for pt_set in all_pts:
#     plt.scatter(*zip(*pt_set))


# img


# import pytesseract as pt

# # from pytesseract import image_to_string
# # from pytesseract import image_to_boxes

# with mss() as sct:
#     time.sleep(1)
#     sct_img = sct.grab(sct.monitors[0])
#     img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

# pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'



