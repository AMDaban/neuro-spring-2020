import numpy as np

class Convolution2D:
    def __init__(self, kernel, stride=1, pad_inp=True):
        k_h, k_w = kernel.shape
        if k_h != k_w:
            raise Exception("Invalid Kernel")

        self._size = k_h
        self._kernel = np.flipud(np.fliplr(kernel))
        self._stride = stride
        self._pad_inp = pad_inp

    def apply(self, inp_array):
        if self._pad_inp:
            inp_array = self.pad_array(inp_array)

        h, w = inp_array.shape

        result_size = (
            int(np.ceil((h - self._size + 1) / self._stride)),
            int(np.ceil((w - self._size + 1) / self._stride))
        )
        result = np.zeros(result_size, dtype=float)
        
        r_i, r_j = -1, -1
        for i in range(0, h - self._size + 1, self._stride):
            r_i += 1
            for j in range(0, w - self._size + 1, self._stride):
                r_j += 1
                result[r_i, r_j] = np.sum(
                    self._kernel * inp_array[i:i + self._size, j:j + self._size]
                )
            r_j = -1

        return result
                

    def pad_array(self, inp_array):
        k_radius = (self._size // 2)

        new_inp_size = (
            2 * k_radius + inp_array.shape[0],
            2 * k_radius + inp_array.shape[1]
        )
        new_array = np.zeros(new_inp_size)
        new_array[k_radius:-k_radius, k_radius:-k_radius] = inp_array

        return new_array
