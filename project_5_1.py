import numpy as np
import cv2
import matplotlib.pyplot as plt

from neurokit.filters.dog import DOG2D
from neurokit.operators.convoloution import Convolution2D

def main():
    img = cv2.imread('pics/car.jpg', 0)

    kernel = DOG2D(3, 0.5, 1).compute()
    operator = Convolution2D(kernel, stride=2)

    result = operator.apply(img)

    plt.imshow(img, "gray")
    plt.show()

    plt.imshow(result, "gray")
    plt.show()

if __name__ == "__main__":
    main()