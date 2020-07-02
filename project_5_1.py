import numpy as np
import cv2
import matplotlib.pyplot as plt

from neurokit.filters.dog import DOG2D
from neurokit.operators.convoloution import Convolution2D

IMAGE_PATH = "pics/pic.jpg"

DOG_SIZE = 7
DOG_SIGMA_1 = 0.3
DOG_SIGMA_2 = 1
DOG_PARTITIONS = 3

CONVOLUTION_PADDING = True
CONVOLUTION_STRIDE = 1
CONVOLUTION_KEEPING_PERCENTAGE = 0.2

def main():
    img = cv2.imread(IMAGE_PATH, 0)

    plt.imshow(img, 'gray')
    plt.show()

    kernel = DOG2D(DOG_SIZE, DOG_SIGMA_1, DOG_SIGMA_2).compute()
    operator = Convolution2D(kernel, stride=CONVOLUTION_STRIDE, pad_inp=CONVOLUTION_PADDING)

    result = operator.apply(img)
    result = normalize_image(result)
    result = result * (result >= CONVOLUTION_KEEPING_PERCENTAGE * np.max(result))

    plt.imshow(result, 'gray')
    plt.show()

    dog_partition_size = 255 / DOG_PARTITIONS
    for partition_idx in range(DOG_PARTITIONS):
        lower_bound = partition_idx * dog_partition_size
        upper_bound = (partition_idx + 1) * dog_partition_size
        partition = result * (lower_bound <= result)
        partition = partition * (partition < upper_bound)
        plt.imshow(partition, 'gray')
        plt.show()

def normalize_image(image):
    unit = (np.max(image) - np.min(image)) / 255
    return (image - np.min(image)) / unit

if __name__ == "__main__":
    main()