# Numpy offers comprehensive mathematical functions in Python
import numpy as np

# OpenCV provides a optimized tools for Computer Vision.
import cv2 as cv

# Matplotlib is a library for creating visualizations in Python.
from matplotlib import pyplot as plt

#Scikit-image offers skeletonize method
import skimage
from skimage import morphology

def erosion_dilation_opening_closing():
    #Load source matrix for morphological operations
    src_mat = np.load('src_mat.npy')

    kernel_4 = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)) #4-neighbor kernel
    kernel_8 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) #8-neighbor kernel

    #Erosion - shrinks the foreground and removes foreground outliers
    eroded_mat = cv.erode(src_mat, kernel_4)

    #Dilation - expands the foreground and fills holes
    dilated_mat = cv.dilate(src_mat, kernel_8)

    #Opening - Erosion followed by a dilation
    opening_mat = cv.morphologyEx(src_mat, cv.MORPH_OPEN, kernel_4)

    #Closing - Dilation followed by an erosion
    closing_mat = cv.morphologyEx(src_mat, cv.MORPH_CLOSE, kernel_8)

    #Plot and save the results
    fig, axes = plt.subplots(nrows=1, ncols=5)
    fig.set_size_inches(20, 10)

    axes[0].matshow(src_mat, cmap=plt.cm.gray)
    axes[0].set_title('source image')
    axes[0].set_axis_off()

    axes[1].matshow(eroded_mat, cmap=plt.cm.gray)
    axes[1].set_title('erosion with N4')
    axes[1].set_axis_off()

    axes[2].matshow(dilated_mat, cmap=plt.cm.gray)
    axes[2].set_title('dilation with N8')
    axes[2].set_axis_off()

    axes[3].matshow(opening_mat, cmap=plt.cm.gray)
    axes[3].set_title('opening with N4')
    axes[3].set_axis_off()

    axes[4].matshow(closing_mat, cmap=plt.cm.gray)
    axes[4].set_title('closing with N8')
    axes[4].set_axis_off()

    plt.tight_layout()
    plt.savefig('output/erosion_dilation_opening_closing.png')
    plt.show()


def road_extraction():
    src_img = cv.imread('./images/street.jpeg')

    #Convert the image to grayscale
    gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    #Apply thresholding to create a binary image
    binary_img = cv.threshold(gray, 94, 255, cv.THRESH_BINARY)[1]

    #Create a kernel for opening operation
    kernel_for_trees = cv.getStructuringElement(cv.MORPH_RECT, (3, 7))

    #Apply opening operation to remove the trees from the image
    opening_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel_for_trees)

    plt.figure(figsize=(16, 8), dpi=91)
    plt.subplot(1, 3, 1)
    plt.imshow(src_img)
    plt.title('source image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.bitwise_not(binary_img), cmap='binary')
    plt.title('binary image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.bitwise_not(opening_img), cmap='binary')
    plt.title('opening result')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/road_extraction.png')
    plt.show()

    return opening_img


def car_road_fill(opening_img):
    src_img = cv.imread('./images/street.jpeg')

    # Convert the image to grayscale
    gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image that only contains the car
    car_img = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)[1]

    # Create a kernel for closing operation on car_img
    kernel_for_car = cv.getStructuringElement(cv.MORPH_RECT, (5, 17))

    # Apply closing operation to fill the car
    car_close = cv.morphologyEx(car_img, cv.MORPH_CLOSE, kernel_for_car)

    # Find the center of the car using moments - a particular weighted average of image pixel intensities
    M = cv.moments(car_close)
    cX = int(M["m10"] / M["m00"]) # Calculate the x-coordinate of the center
    cY = int(M["m01"] / M["m00"]) # Calculate the y-coordinate of the center
    car_width = cv.countNonZero(car_close[cY][:])
    car_height = cv.countNonZero(car_close[:, cX])

    # kernel used in closing operation on opening_img (image with trees removed)
    kernel_to_fill = cv.getStructuringElement(cv.MORPH_RECT, (car_width, car_height))

    # Apply closing operation to fill the road
    closing_img = cv.morphologyEx(opening_img, cv.MORPH_CLOSE, kernel_to_fill)

    plt.figure(figsize=(16, 10), dpi=91)
    plt.subplot(2, 2, 1)
    plt.imshow(np.bitwise_not(car_img), cmap='binary')
    plt.title('binary image of the car')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(np.bitwise_not(car_close), cmap='binary')
    plt.title('closing result of the car')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(np.bitwise_not(opening_img), cmap='binary')
    plt.title('opening result')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(np.bitwise_not(closing_img), cmap='binary')
    plt.title('closing result after opening')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/car_road_fill.png')
    plt.show()

    return closing_img

def road_edge(closing_img):
    kernel_8 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    #Erosion of the filled road image
    eroded_closing_img = cv.erode(closing_img, kernel_8)

    #Dilation of the filled road image
    dilated_closing_img = cv.dilate(closing_img, kernel_8)

    #Edge = Dilation result minus erosion result (Outer boundary and Inner boundary)
    edge_img = dilated_closing_img - eroded_closing_img

    #Alternatively, the edge can be obtained using the morphological gradient openCV function
    # edge_img = cv.morphologyEx(closing_img, cv.MORPH_GRADIENT, kernel_8)

    #Plot and save the results
    plt.figure(figsize=(16, 10), dpi=91)

    plt.subplot(1, 2, 1)
    plt.imshow(np.bitwise_not(closing_img), cmap='binary')
    plt.title('the whole road')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(np.bitwise_not(edge_img), cmap='binary')
    plt.title('edge of the road')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/road_edge.png')
    plt.show()

    return edge_img

def dist_transform_skeletonize(closing_img):
    #Distance transformation of the closing image - obtain the derived representation of a binary image,
    # where the value of each pixel is replaced by its distance to the nearest background pixel
    dist = cv.distanceTransform(closing_img, cv.DIST_L2, 3)

    #Normalize the distance image into range (0,1)
    dist_norm = cv.normalize(dist, 0, 255, cv.NORM_MINMAX)

    #Threshold the distance image to get the binary image
    dist_binary = cv.threshold(dist_norm, 0.7, 1.0, cv.THRESH_BINARY)[1]

    #Skeletonize the binary image to get the center line of the road - reduces binary objects to 1px wide representations
    middle_line = skimage.morphology.skeletonize(dist_binary, method='lee')

    #Plot and save the results
    plt.figure(figsize=(16, 10), dpi=91)
    plt.subplot(1, 2, 1)
    plt.imshow(1.0 - dist_binary, cmap='binary')
    plt.title('distance image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(1.0 - middle_line, cmap='binary')
    plt.title('middle line of the road')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/road_center.png')
    plt.show()

    return middle_line

def morph_result(edge_img, middle_line):
    plt.figure(figsize=(16, 10), dpi=300)

    plt.subplot(1, 2, 1)
    src_img = cv.imread('./images/street.jpeg')
    plt.imshow(src_img)
    plt.axis('off')
    plt.title('source image')

    # Overlay the edge and middle line on the source image
    result_img = src_img.copy()
    result_img[edge_img == 255] = [0, 0, 255]
    result_img[middle_line == 255] = [255, 0, 0]

    plt.subplot(1, 2, 2)
    plt.imshow(result_img)
    plt.axis('off')
    plt.title('result image')

    plt.tight_layout()
    plt.savefig('output/morph_result.png')
    plt.show()