# Numpy offers comprehensive mathematical functions in Python
import numpy as np

# OpenCV provides a optimized tools for Computer Vision.
import cv2 as cv

# Matplotlib is a library for creating visualizations in Python.
from matplotlib import pyplot as plt

# OS is a library for interacting with the operating system.
import os

#Task 1: Load and visualize one color image in BGR color scale, the gray value image, and the image in RGB color scale
def image_io():
    I = cv.imread('images/image.jpg')
    I_gray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
    I_RGB = cv.cvtColor(I, cv.COLOR_BGR2RGB)
    os.makedirs('output', exist_ok=True)
    cv.imwrite('output/gray.png', I_gray)

    plt.figure(figsize=(16, 8), dpi=96)
    plt.subplot(1, 3, 1)
    plt.imshow(I)
    plt.title('BGR Image', fontsize=40)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(I_RGB)
    plt.title('RGB Image', fontsize=40)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(I_gray, cmap='gray')
    plt.title('Gray Image', fontsize=40)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/bgr_rgb_gray.png')
    plt.show()

    return I, I_gray, I_RGB

#Task 2: Spilt individual channels from one color image (i.e. R, G, B), and then merge them back to one single image in a different order (G, B, R)
def spilt_and_merge(I_RGB):
    r, g, b = cv.split(I_RGB) # Split the image into its channels
    I_merged = cv.merge((g, b, r)) # Merge the channels in a different order

    plt.figure(figsize=(45, 8), dpi=96) #Visualize the results

    plt.subplot(1, 4, 1)
    plt.imshow(r, cmap='gray')
    plt.title('Red', fontsize=40)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(g, cmap='gray')
    plt.title('Green', fontsize=40)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(b, cmap='gray')
    plt.title('Blue', fontsize=40)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(I_merged)
    plt.title('Merge Image', fontsize=40)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/r_g_b_merge.png')
    plt.show()

    return r, g, b, I_merged


#Task 3: Point operation - apply the point operation and create a image with inverted color.
def invert_color(image):
    I_inverse = 255 - image
    plt.figure(figsize=(20, 16), dpi=68)

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Source image', fontsize=30)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(I_inverse)
    plt.title('After color inversion', fontsize=30)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/inverted.png')
    plt.show()
    return I_inverse


#Task 4: Statistical image characterization - calculate the mean and variance of the gray image and also for the RGB channels individually.
def image_statistics(I_gray, r, g, b):

    #Compute mean and standard deviation for each channel
    mean_gray, std_gray = cv.meanStdDev(I_gray) #grayscale image
    mean_red, std_red = cv.meanStdDev(r) #red channel
    mean_green, std_green = cv.meanStdDev(g) #green channel
    mean_blue, std_blue = cv.meanStdDev(b) #blue channel

    #Collect results in arrays
    collect_mean = np.array([mean_gray, mean_red, mean_green, mean_blue]).reshape(1, 4)
    collect_std = np.array([std_gray, std_red, std_green, std_blue]).reshape(1, 4)
    collect_variance = np.sqrt(collect_std)

    print("Mean of images (Gray Image, Red, Green, Blue):", collect_mean)
    print("Variance of images (Gray Image, Red, Green, Blue):", collect_variance)

    return collect_mean, collect_variance

#Task 5: Calculate the histogram of gray scale images and observe their differences for over-exposure and under-exposure images.
def histogram_exposure(I_RGB, I_gray):
    #Load images
    I_over = cv.imread("images/image_exposure1.jpg")
    I_RGB_over = cv.cvtColor(I_over, cv.COLOR_BGR2RGB)
    I_under = cv.imread("images/image_exposure2.jpg")
    I_RGB_under = cv.cvtColor(I_under, cv.COLOR_BGR2RGB)

    #Compute histograms for each image
    hist_gray = cv.calcHist([I_gray], [0], None, [256], [0, 255]) / I_gray.size
    hist_over = cv.calcHist([I_over], [0], None, [256], [0, 255]) / I_over.size
    hist_under = cv.calcHist([I_under], [0], None, [256], [0, 255]) / I_under.size

    # Histogram for a color image with respect to individual channels
    channel = cv.split(I_RGB) # Split the image into its channels
    hist_r = cv.calcHist([channel[0]], [0], None, [256], [0, 255])
    hist_g = cv.calcHist([channel[1]], [0], None, [256], [0, 255])
    hist_b = cv.calcHist([channel[2]], [0], None, [256], [0, 255])

    #Visualize and save the results
    plt.figure(figsize=(25, 8), dpi=100)

    plt.subplot(1, 3, 1)
    plt.imshow(I_RGB)
    plt.title('Normal', fontsize=20)

    plt.subplot(1, 3, 2)
    plt.imshow(I_RGB_over)
    plt.title('Overexposure', fontsize=20)

    plt.subplot(1, 3, 3)
    plt.imshow(I_RGB_under)
    plt.title('Underexposure', fontsize=20)

    plt.savefig('output/normal_over_under.png')
    plt.show()

    plt.figure(figsize=(25, 8), dpi=150)

    plt.subplot(1, 3, 1)
    plt.plot(hist_gray)
    plt.xlim([0, 255])
    plt.title('Normal', fontsize=20)

    plt.subplot(1, 3, 2)
    plt.plot(hist_over)
    plt.xlim([0, 255])
    plt.title('Overexposure', fontsize=20)

    plt.subplot(1, 3, 3)
    plt.plot(hist_under)
    plt.xlim([0, 255])
    plt.title('Underexposure', fontsize=20)

    plt.tight_layout()
    plt.savefig('output/histogram.png')
    plt.show()

    plt.figure(figsize=(8, 4), dpi=150)
    plt.plot(hist_r[:, 0], color='r')
    plt.plot(hist_g[:, 0], color='g')
    plt.plot(hist_b[:, 0], color='b')
    plt.xlim([0, 255])
    plt.title('Histogram of images for seperate channels', fontsize=20)

    plt.tight_layout()
    plt.savefig('output/histogram_color.png')
    plt.show()

    return hist_gray, hist_over, hist_under, hist_r, hist_g, hist_b