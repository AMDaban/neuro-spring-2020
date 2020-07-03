import numpy as np
import cv2
import matplotlib.pyplot as plt

from neurokit.filters.gabor import Gabor2D
from neurokit.operators.convoloution import Convolution2D

IMAGE_PATH = "pics/texture.jpg"

GABOR_SIZE = 37
GABOR_LAMBDA = 10
GABOR_THETA = 1/2 * np.pi 
GABOR_SIGMA = 3
GABOR_GAMMA = 0.5
GABOR_PARTITIONS = 5

CONVOLUTION_PADDING = True
CONVOLUTION_STRIDE = 1
CONVOLUTION_KEEPING_PERCENTAGE = 0.2

def main():
    img = cv2.imread(IMAGE_PATH, 0) 

    plt.imshow(img, 'gray')
    plt.show()

    kernel = Gabor2D(GABOR_SIZE, GABOR_LAMBDA, GABOR_THETA, GABOR_SIGMA, GABOR_GAMMA).compute()
    operator = Convolution2D(kernel, stride=CONVOLUTION_STRIDE, pad_inp=CONVOLUTION_PADDING)

    result = operator.apply(img)
    result = normalize_image(result)
    result = result * (result >= CONVOLUTION_KEEPING_PERCENTAGE * np.max(result))

    plt.imshow(result, 'gray')
    plt.show()

    gabor_partition_size = 255 / GABOR_PARTITIONS
    for partition_idx in range(GABOR_PARTITIONS):
        lower_bound = partition_idx * gabor_partition_size
        upper_bound = (partition_idx + 1) * gabor_partition_size
        partition = result * (lower_bound <= result)
        partition = partition * (partition < upper_bound)
        plt.imshow(partition, 'gray')
        plt.show()

def normalize_image(image):
    unit = (np.max(image) - np.min(image)) / 255
    return (image - np.min(image)) / unit

if __name__ == "__main__":
    main()