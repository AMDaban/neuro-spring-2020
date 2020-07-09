import numpy as np
import cv2
import matplotlib.pyplot as plt

from neurokit.filters.dog import DOG2D
from neurokit.operators.convoloution import Convolution2D

IMAGE_PATH = "pics/einstein.jpg"

def main():
    main_single(dog_size=7,dog_sigma_1=0.3,dog_sigma_2=1,dog_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0.0)
    main_single(dog_size=11,dog_sigma_1=0.3,dog_sigma_2=1,dog_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0.0)
    main_single(dog_size=15,dog_sigma_1=0.3,dog_sigma_2=1,dog_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0.0)
    main_single(dog_size=19,dog_sigma_1=0.3,dog_sigma_2=1,dog_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0.0)
    main_single(dog_size=23,dog_sigma_1=0.3,dog_sigma_2=1,dog_partitions=3,conv_padding=True,conv_stride=1,conv_kp=0.0)

def main_single(dog_size, dog_sigma_1, dog_sigma_2, dog_partitions, conv_padding, conv_stride, conv_kp):
    img = cv2.imread(IMAGE_PATH, 0)

    plt.imshow(img, 'gray')
    plt.show()

    kernel = DOG2D(dog_size, dog_sigma_1, dog_sigma_2).compute()
    operator = Convolution2D(kernel, stride=conv_stride, pad_inp=conv_padding)

    result = operator.apply(img)
    result = normalize_image(result)
    result = result * (result >= conv_kp * np.max(result))

    plt.imshow(result, 'gray')
    plt.show()

    dog_partition_size = 255 / dog_partitions
    for partition_idx in range(dog_partitions):
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