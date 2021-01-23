import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from shapely.geometry import Polygon
import imutils
import PIL

import rawpy
import tqdm
import sklearn

DPI = 1600

film_format = {
    "35": {"width": 36.0, "height": 24.0},
    "6x4.5": {"width": 56.0, "height": 42.0},
    "6x6": {"width": 56.0, "height": 56.0},
    "6x7": {"width": 56.0, "height": 67.0},
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


def detect(
    im,
    fmt="35",
    area_threshold=0.1,
    aspect_ratio_threshold=0.1,
    dpi=1600,
    to_bw=True,
    thresholds=np.linspace(200, 100, 11),
    use_tqdm=False,
    verbose=False,
):
    polys = []
    areas = []
    paths = []
    rotations = []

    target_area = film_area(fmt, dpi)
    target_aspect_ratio = film_format[fmt]["width"] / film_format[fmt]["height"]

    bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if verbose:
        fig, axs = plt.subplots(len(thresholds), 1, figsize=(20, 200))

    if use_tqdm:
        thresholds = tqdm.tqdm(thresholds)

    for i, threshold in enumerate(thresholds):
        curr_paths = []
        ret, threshed = cv2.threshold(bw, threshold, 255, cv2.THRESH_BINARY)

        cnts = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            rect = cv2.minAreaRect(c)
            (x, y), (w, h), rotation = rect
            if w == 0 or h == 0:
                continue
            box = cv2.boxPoints(rect)
            area = w * h
            aspect_ratio = max(h, w) / min(h, w)

            if verbose:
                if area > 1000000:
                    print(area)

            if (
                area < target_area * (1 - area_threshold)
                or area > target_area * (1 + area_threshold)
                or aspect_ratio < target_aspect_ratio * (1 - aspect_ratio_threshold)
                or aspect_ratio > target_aspect_ratio * (1 + aspect_ratio_threshold)
            ):
                continue

            currPoly = Polygon(box)
            box = np.concatenate((box, [box[0]]))
            exists = False
            curr_paths.append(box)
            for j, poly in enumerate(polys):
                intersected_area = poly.intersection(currPoly).area
                if intersected_area > 0.9 * areas[i]:
                    exists = True
                    if areas[i] < area:
                        polys[j] = currPoly
                        areas[j] = area
                        rotations[j] = rotations
                        paths[j] = box
                    break

            if exists:
                continue

            polys.append(currPoly)
            areas.append(area)
            rotations.append(rotation)
            paths.append(box)

        if verbose:
            show(threshed, curr_paths, ax=axs[i])

    return paths, rotations


def show(im, paths, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(im, cmap="gray")

    for path in paths:
        path = path.squeeze()
        codes = [Path.MOVETO]
        codes.extend([Path.LINETO] * 3)
        codes.append(Path.CLOSEPOLY)
        path = Path(path, codes)
        patch = patches.PathPatch(path)
        ax.add_patch(patch)


def crop(im, paths, width=None, height=None, pad=0.25, dpi=1600):
    padding = int(pad * dpi)

    cropped = []
    for path in paths:
        x, y, w, h = cv2.boundingRect(path)
        if width is not None:
            pixel_width = int(width * dpi)
            x -= (pixel_width - h) // 2
            h = pixel_width
        if height is not None:
            pixel_height = int(height * dpi)
            y -= (pixel_height - w) // 2
            w = pixel_height
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(x + w + 2 * padding, im.shape[1]) - x
        h = min(y + h + 2 * padding, im.shape[0]) - y
        cropped.append(im[y : y + h, x : x + w])

    return cropped


def rotate(cropped, rotations):
    rotated = []
    for crop, rotation in zip(cropped, rotations):
        rotated.append(imutils.rotate_bound(crop, -rotation * 180 / np.pi))

    return rotated


def save(ims, directory, format="tiff"):
    directory = os.path.realpath(os.path.expanduser(directory))

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, im in enumerate(ims):
        im = PIL.Image.fromarray(im)
        name = f"{directory}/{i:05d}.{format}"
        im.save(name, format=format, quality=100, compression=None)

