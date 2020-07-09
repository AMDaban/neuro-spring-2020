import numpy as np
import cv2
import matplotlib.pyplot as plt

from neurokit.filters.gabor import Gabor2D
from neurokit.operators.convoloution import Convolution2D

IMAGE_PATH = "pics/einstein.jpg"

def main():
    main_single(g_size=3,g_lambda=15,g_sigma=7.5,g_gamma=0.5,g_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0)
    main_single(g_size=5,g_lambda=15,g_sigma=7.5,g_gamma=0.5,g_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0)
    main_single(g_size=7,g_lambda=15,g_sigma=7.5,g_gamma=0.5,g_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0)
    main_single(g_size=9,g_lambda=15,g_sigma=7.5,g_gamma=0.5,g_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0)
    main_single(g_size=11,g_lambda=15,g_sigma=7.5,g_gamma=0.5,g_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0)
    main_single(g_size=13,g_lambda=15,g_sigma=7.5,g_gamma=0.5,g_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0)
    main_single(g_size=15,g_lambda=15,g_sigma=7.5,g_gamma=0.5,g_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0)
    main_single(g_size=17,g_lambda=15,g_sigma=7.5,g_gamma=0.5,g_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0)


def main_single(g_size, g_lambda, g_sigma, g_gamma, g_partitions, conv_padding, conv_stride, conv_kp):
    img = cv2.imread(IMAGE_PATH, 0) 

    plt.imshow(img, 'gray')
    plt.show()

    for i in range(4):
        theta = i * (np.pi)/4
        kernel = Gabor2D(g_size, g_lambda, theta, g_sigma, g_gamma).compute()
        operator = Convolution2D(kernel, stride=conv_stride, pad_inp=conv_padding)

        result = operator.apply(img)
        result = normalize_image(result)
        result = result * (result >= conv_kp * np.max(result))

        plt.imshow(result, 'gray')
        plt.show()

        gabor_partition_size = 255 / g_partitions
        for partition_idx in range(g_partitions):
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