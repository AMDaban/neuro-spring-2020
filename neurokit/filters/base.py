import numpy as np

class Filter:
    def __init__(self, size):
        self._size = size

    def _index_value_func(self):
        raise Exception('not implemented')

    def compute(self):
        kernel = np.fromfunction(self._index_value_func(), (self._size, self._size))
        return kernel - np.mean(kernel)