# Numpy offers comprehensive mathematical functions in Python
import numpy as np

# OpenCV provides a optimized tools for Computer Vision.
import cv2 as cv

# Matplotlib is a library for creating visualizations in Python.
from matplotlib import pyplot as plt

#Scikit-image offers a random noise method
from skimage.util import random_noise

def padding_convolution():
    src_img = cv.imread('./images/house.jpg', cv.IMREAD_GRAYSCALE)

    #Define a kernel
    kernel = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3]
    ])

    #Apply padding
    img = cv.copyMakeBorder(src_img, 20, 20, 20, 20, cv.BORDER_CONSTANT, None, value=0)
    zero_padding = cv.copyMakeBorder(src_img, 20, 20, 20, 20, cv.BORDER_CONSTANT) #Adds a constant colored border
    wrap_padding = cv.copyMakeBorder(src_img, 20, 20, 20, 20, cv.BORDER_WRAP) #wraps the image by repeating the border elements
    clamp_padding = cv.copyMakeBorder(src_img, 20, 20, 20, 20, cv.BORDER_REPLICATE) #last element is replicated
    mirror_padding = cv.copyMakeBorder(src_img, 20, 20, 20, 20, cv.BORDER_REFLECT) #border will be mirror reflection of the border elements

    #Apply convolution
    conv_mat = cv.filter2D(src_img, -1, kernel) #-1 means the output image will have the same depth as the source image

    #Visualize the results
    plt.figure(figsize=(18, 8), dpi=96)
    plt.subplot(1, 6, 1)
    plt.title("source image")
    plt.imshow(src_img, cmap='gray')

    plt.subplot(1, 6, 2)
    plt.title("zero padding")
    plt.imshow(zero_padding, cmap=plt.cm.gray)

    plt.subplot(1, 6, 3)
    plt.title("wrap padding")
    plt.imshow(wrap_padding, cmap=plt.cm.gray)

    plt.subplot(1, 6, 4)
    plt.title("clamp padding")
    plt.imshow(clamp_padding, cmap=plt.cm.gray)

    plt.subplot(1, 6, 5)
    plt.title("mirror padding")
    plt.imshow(mirror_padding, cmap=plt.cm.gray)

    plt.subplot(1, 6, 6)
    plt.title("convolution result")
    plt.imshow(conv_mat, cmap='gray')

    plt.tight_layout()
    plt.savefig('output/padding_convolution.png')
    plt.show()

def box_binomial_median_filters():
    src_img = cv.imread('./images/house.jpg', cv.IMREAD_GRAYSCALE)

    #Box filter
    box_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5)) / 25
    box_result = cv.filter2D(src_img, -1, box_kernel) #convolution result using box filter

    #Binomial filter
    binomial_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16

    binomial_result = cv.filter2D(src_img, -1, binomial_kernel) #convolution result using binomial filter

    #Median filter for salt and pepper noise
    saltNpepper_noise_img = (255 * random_noise(src_img, mode='s&p')).astype(np.uint8)
    median_result = cv.medianBlur(saltNpepper_noise_img, 5) #convolution result using median filter

    #Visualize and save the results
    plt.figure(figsize=(6, 9), dpi=109)
    plt.subplot(3, 2, 1)
    plt.imshow(src_img, cmap='gray')
    plt.title("source image")
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(box_result, cmap='gray')
    plt.title("box filter")
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.imshow(src_img, cmap='gray')
    plt.title("source image")
    plt.axis('off')

    plt.subplot(3, 2, 4)
    plt.imshow(binomial_result, cmap='gray')
    plt.title("binomial filter")
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.imshow(saltNpepper_noise_img, cmap='gray')
    plt.title("image with B/W noise")
    plt.axis('off')

    plt.subplot(3, 2, 6)
    plt.imshow(median_result, cmap='gray')
    plt.title("median filter")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/box_binomial_median_filters.png')
    plt.show()

    return box_result, binomial_result, median_result


def sobel_filter():
    src_img = cv.imread('./images/house.jpg', cv.IMREAD_GRAYSCALE)

    #kernel of sobel filter in x direction
    sobel_x_kernel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]) / 8

    #kernel of sobel filter in y direction
    sobel_y_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]) / 8

    gradient_x = cv.filter2D(src_img, cv.CV_64F, sobel_x_kernel) #image gradient in x direction
    gradient_y = cv.filter2D(src_img, cv.CV_64F, sobel_y_kernel) #image gradient in y direction

    #Alternative way to calculate the gradient in x and y direction with cv.Sobel
    # gradient_x = cv.Sobel(src_img, cv.CV_64F, dx=1, dy=0, ksize=3)
    # gradient_y = cv.Sobel(src_img, cv.CV_64F, dx=0, dy=1, ksize=3)

    #Combining this two we can identify parts of the image that look like edges
    magnitude = np.sqrt((gradient_x ** 2) + (gradient_y ** 2)) # magnitude of the gradient shows how strongly the intensity changes
    direction = np.arctan2(gradient_y, gradient_x) #shows the angle of the gradient

    #Visualize and save the results
    plt.figure(figsize=(6, 6), dpi=109)
    plt.subplot(2, 2, 1)
    plt.imshow(gradient_x, cmap='gray')
    plt.title("sobel_gradient_x")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(magnitude, cmap='Greys')
    plt.title("magnitude")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(gradient_y, cmap='gray')
    plt.title("sobel_gradient_y")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    direction[magnitude < 5] = np.nan
    plt.imshow(direction, cmap='hsv')
    plt.title("direction")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/sobel_filter.png')
    plt.show()

def laplacian_filter(binomial_result):
    src_img = cv.imread('./images/house.jpg', cv.IMREAD_GRAYSCALE)

    #Kernel of laplacian filter
    laplacian_kernel_4 = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    #Apply laplacian filter - increased noise level (interprets image noise as an edge)
    lap = cv.filter2D(src_img, -1, laplacian_kernel_4)

    #Apply laplacian filter on the smoothed image after binomial filter - clear edges in the final result
    lap_binomial = cv.filter2D(binomial_result, -1, laplacian_kernel_4)

    #Visualize and save the results
    plt.figure(figsize=(6, 3), dpi=109)
    plt.subplot(1, 2, 1)
    plt.imshow(cv.convertScaleAbs(lap), cmap='gray')
    plt.axis('off')
    plt.title("Laplacian 4")

    plt.subplot(1, 2, 2)
    plt.imshow(cv.convertScaleAbs(lap_binomial), cmap='gray')
    plt.axis('off')
    plt.title("Laplacian 4 on binomial")

    plt.tight_layout()
    plt.savefig('output/laplacian_filter.png')
    plt.show()