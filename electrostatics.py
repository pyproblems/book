# -*- coding: utf-8 -*-
"""This module defines some essential functions for the two notebooks."""
import time

import numpy as np

k_e = 1 / (4 * np.pi * 8.8541878128e-12)


def E(x, y, q, r):
    """Electric field.

    Args:
        x (float): X position(s).
        y (float): Y position(s).
        q (float): Charge(s).
        r (iterable of float): (x, y) position(s) of the point charge(s). If an array is given,
            it should be a (2, N) array where N is the number of point charges.

    Returns:
        float: Electric field vectors at every point in `x` and `y`. The shape of
        this array is the same shape as `x` and `y` with an added initial dimension.

    """
    # Calculate the distance of each requested point from the point charge.
    d = ((x - r[0]) ** 2 + (y - r[1]) ** 2) ** 0.5
    magnitudes = k_e * q / d ** 2

    # Calculate unit vector components.
    xs = (x - r[0]) / d
    ys = (y - r[1]) / d

    return np.concatenate(
        ((xs * magnitudes)[np.newaxis], (ys * magnitudes)[np.newaxis]), axis=0
    )


def E_dir(x, y, q, r):
    """Electric field direction at one point (x, y).

    Args:
        x (float): x position.
        y (float): y position.
        q (float): Charge(s).
        r (iterable of float): (x, y) position(s) of the point charge(s). If an array is given,
            it should be a (2, N) array where N is the number of point charges.

    Returns:
        float: Normalised electric field vectors at every point in `x` and `y`. The shape of
        this array is the same shape as `x` and `y` with an added initial dimension.

    """
    E_field = np.sum(E(x, y, q, r), axis=1)
    # Normalise the electric field vectors.
    return E_field / np.linalg.norm(E_field)


class Timer:
    """Can be used as a context manager to time events."""

    def __init__(self, name="operation"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

        print(f"{self.name} took {self.interval:0.1e} seconds.")
