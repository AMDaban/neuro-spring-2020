import numpy as np

from .base import Filter

class Gabor2D(Filter):
    def __init__(self, size, lmbda, theta, sigma, gamma):
        Filter.__init__(self, size)

        self._lambda = lmbda
        self._theta = theta
        self._sigma = sigma
        self._gamma = gamma

    def _index_value_func(self):
        radius = self._size // 2
        def index_value(i, j):
            i -= radius
            j -= radius

            x = i * np.cos(self._theta) + j * np.sin(self._theta)
            y = -j * np.sin(self._theta) + j * np.cos(self._theta)

            part_1 = np.exp(-(x ** 2 + self._gamma ** 2 * y ** 2) / (2 * self._sigma ** 2))
            part_2 = np.cos(2 * np.pi * x / self._lambda)
            return part_1 * part_2
        return index_value