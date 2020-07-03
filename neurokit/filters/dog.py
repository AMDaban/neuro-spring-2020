import numpy as np

from .base import Filter

class DOG2D(Filter):
    def __init__(self, size, s1, s2):
        Filter.__init__(self, size)

        self._s1 = s1
        self._s2 = s2

    def _index_value_func(self):
        radius = self._size // 2
        def index_value(i, j):
            i -= radius
            j -= radius

            g1 = (1 / self._s1) * np.exp(-(i ** 2 + j ** 2) / (2 * self._s1 ** 2))
            g2 = (1 / self._s2) * np.exp(-(i ** 2 + j ** 2) / (2 * self._s2 ** 2))
            return (1 / np.sqrt(2 * np.pi) * (g1 - g2))
        return index_value