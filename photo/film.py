import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from shapely.geometry import Polygon
import imutils

import rawpy
import tqdm
import sklearn

DPI = 1600

film_format = {
    "35": {
        "width": 36.,
        "height": 24.
    },
    "6x4.5": {
        "width": 56.,
        "height": 42.
    },
    "6x6": {
        "width": 56.,
        "height": 56.
    },
    "6x7": {
        "width": 56.,
        "height": 67.
    }
}


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def pixel_area(width, height, dpi):
    return width / 25.4 * dpi * height / 25.4 * dpi


def l2(v1, v2):
    return np.linalg.norm(v1 - v2)


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def film_area(fmt, dpi):
    fmt = film_format[fmt]
    return pixel_area(fmt["width"], fmt["height"], dpi)


def detect(im, fmt="35", area_threshold=0.1, dpi=1600):
    polys = []
    paths = []
    bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    target_area = film_area(fmt, dpi)
    for threshold in np.linspace(200, 100, 11):
        ret, threshed = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)
        cnts = cv2.findContours(threshed.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in tqdm.tqdm(cnts):
            idxs = cv2.convexHull(c, returnPoints=False)
            hull = c[idxs.squeeze()]
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.015 * peri, True)

            if len(approx) == 4:
                verts = approx.squeeze()
                area = poly_area(verts[:, 0], verts[:, 1])

                if area < target_area * (1 - area_threshold) or area > target_area * (1 + area_threshold):
                    continue

                currPoly = Polygon(verts)
                exists = False
                for poly in polys:
                    if poly.intersects(currPoly):
                        exists = True
                        break

                if exists:
                    continue

                polys.append(currPoly)

                verts = np.concatenate((verts, [verts[0]]))
                rect = cv2.minAreaRect(verts)
                box = cv2.boxPoints(rect)
                box = np.int0(np.concatenate((box, [box[0]])))

                paths.append(box)

    return paths


def show(im, paths):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(im, cmap="gray")

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    for path in paths:
        path = Path(path, codes)
        patch = patches.PathPatch(path)
        ax.add_patch(patch)
