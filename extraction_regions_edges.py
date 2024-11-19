# Numpy offers comprehensive mathematical functions in Python
import numpy as np

# OpenCV provides a optimized tools for Computer Vision.
import cv2 as cv

# Matplotlib is a library for creating visualizations in Python.
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton

#Scikit-image offers segmentation method
import skimage
import skimage.segmentation

def region_growing():
    src_img = cv.imread('./images/extraction/image.tif') #river image
    gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY) #convert to gray scale
    rgb_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB) #convert to RGB

    #Initialize the river region - uint8 ndarray with same shape as src_img, river = 255, non-river = 0
    river_1 = np.zeros((src_img.shape[0], src_img.shape[1]))

    #Define a kernel for morphological closing
    kernel = np.ones((5, 5), np.uint8)

    #Choose reasonable seeds in the river region from the image
    seeds = [(347, 99), (171, 197), (629, 225)] #seed array, each element looks like (row,col)

    for seed in seeds:
        #draw a red circle around the seed
        rgb_img = cv.circle(rgb_img, (seed[1], seed[0]), 5, [255, 0, 0])

        #region growing using flood fill
        mask = skimage.segmentation.flood(gray_img, seed, tolerance=8)
        river_1[mask] = 255

    #morphological closing to fill the holes in river
    river_1 = cv.morphologyEx(river_1.astype(np.uint8), cv.MORPH_CLOSE, kernel)

    #Plot and save the results
    plt.figure(figsize=(12, 6), dpi=96)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(river_1, cmap='binary')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/region_growing.png')
    plt.show()

    return river_1


def watershed(river_1):
    src_img = cv.imread('./images/extraction/image.tif')  # river image
    rgb_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)  # convert to RGB

    #Step 1: Get the middle line using distance transform and skeletonization
    distance_transform = cv.distanceTransform(river_1, cv.DIST_L2, 5) #compute the distance transformation
    distance_transform_normal = cv.normalize(distance_transform, None) #normalize the distance transformation

    #threshold the distance transformation to get the binary image
    _, distance_transform_binary = cv.threshold(distance_transform_normal, 0.004, 1, cv.THRESH_BINARY)

    #skeletonize the binary image to get the middle line of the river
    middle_line_binary = skimage.morphology.skeletonize(distance_transform_binary, method='lee') * 255

    #Step 2: Merge the riverbanks and middle line
    riverbanks_binary = 255 - river_1 #compute the riverbanks
    merged_result = (riverbanks_binary + middle_line_binary).astype(np.uint8)

    #Step 3: Label the binary image using connected components
    _, labels = cv.connectedComponents(merged_result, connectivity=8)
    labels += 1 #add 1 to make sure the background is not labeled as 0

    #Step 4: Apply watershed segmentation
    river_2 = cv.watershed(src_img, labels)

    #Plot and save the results
    plt.figure(figsize=(12, 6), dpi=96)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.axis('off')

    #Watershed result overlay
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_img)
    plt.imshow(river_2, cmap=plt.cm.Set1_r, alpha=.5)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/watershed.png')
    plt.show()


def mean_shift():
    src_img = cv.imread('./images/extraction/image2.tif')
    rgb_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)

    spatialRad = 30 #spatial window radius
    colorRad = 20 #color window radius
    maxPyrLevel = 5 #maximum pyramid level

    #Apply the mean shift filter
    result = cv.pyrMeanShiftFiltering(rgb_img, sp=spatialRad, sr=colorRad, maxLevel=maxPyrLevel)

    plt.figure(figsize=(16, 8), dpi=96)
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title('original image')
    plt.axis('off')

    #Mean shift result
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title('mean shift result')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/mean_shift.png')
    plt.show()

def slic_superpixels():
    src_img = cv.imread('./images/extraction/image2.tif')
    rgb_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)

    #Apply SLIC superpixel segmentation
    #region_size - average superpixel size in pixels, ruler - smoothness factor of superpixel
    superpixels = cv.ximgproc.createSuperpixelSLIC(rgb_img, region_size=30, ruler=10)
    superpixels.iterate(num_iterations=5) #iterate 5 times
    superpixels.enforceLabelConnectivity(min_element_size=20) #enforce label connectivity - superpixels will be connected
    rgb_img[superpixels.getLabelContourMask() == 255] = [255, 0, 0] #draw the superpixel boundaries

    #Plot and save the results
    plt.figure(figsize=(6, 6), dpi=96)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title('SLIC superpixels')
    plt.tight_layout()
    plt.savefig('output/slic_superpixels.png')
    plt.show()

def douglas_peuke():
    src_img = cv.imread('./images/extraction/image3.tif')
    rgb_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    result_img = rgb_img.copy() #copy the rgb image

    #Initialize the river region
    river_3 = np.zeros((src_img.shape[0], src_img.shape[1]))

    #Define a kernel for morphological closing
    kernel = np.ones((5, 5), np.uint8)

    seeds = [(145, 61), (361, 240)]
    for seed in seeds:
        #draw a red circle around the seed
        rgb_img = cv.circle(rgb_img, (seed[1], seed[0]), 5, [255, 0, 0])

        #region growing using flood fill
        mask = skimage.segmentation.flood(gray_img, seed, connectivity=1, tolerance=15)
        river_3[mask] = 255

    # morphological closing to fill the holes in river
    river_3 = cv.morphologyEx(river_3.astype(np.uint8), cv.MORPH_CLOSE, kernel)

    #Apply Douglas Peucker algorithm to simplify the river contour
    contours, hierarchy = cv.findContours(river_3, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #find the contours
    douglas_peuke_result = cv.approxPolyDP(contours[0], epsilon=4, closed=True) #apply Douglas Peucker algorithm
    cv.drawContours(rgb_img, contours, contourIdx=-1, color=(0, 255, 0), thickness=3) #draw the original contour
    cv.drawContours(result_img, [douglas_peuke_result], contourIdx=-1, color=(255, 0, 0), thickness=3) #draw the simplified contour

    #Plot and save the results
    plt.figure(figsize=(12, 6), dpi=96)
    plt.subplot(1, 2, 1)
    plt.title('Original Contour')
    plt.imshow(rgb_img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Douglas Peuke Result')
    plt.imshow(result_img)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/douglas_peuke.png')
    plt.show()


def hough_transform():
    src_img = cv.imread('./images/extraction/image4.tif')
    rgb_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)

    #Define a kernel for morphological opening
    kernel = np.ones((5, 5), np.uint8)

    #Apply thresholding to create a binary image
    _, binary_img = cv.threshold(gray_img, 190, 1, cv.THRESH_BINARY)

    #Apply morphological opening to remove the noise
    binary_img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

    #Find the edges using contour detection
    edges = np.zeros(binary_img.shape) #create a zero array with the same shape as binary_img
    contours, hierarchy = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #Find contours
    cv.drawContours(edges, contours, contourIdx=-1, color=1, thickness=3) #draw the contours on the edges array

    #Apply Hough Circle Transform
    hough_radii = [36]  # radius of the circle
    hough = skimage.transform.hough_circle(edges, hough_radii)

    #find the center of four small fountains (peaks in the hough space)
    accums, cx, cy, radii = skimage.transform.hough_circle_peaks(hough, hough_radii)

    #Draw the detected circles on the image, indicating the position of founded fountains
    for i in range(len(accums)):
        if (accums[i] > 0.7):
            rgb_img = cv.circle(rgb_img, (cx[i], cy[i]), radius=radii[i], color=(255, 0, 0), thickness=3)

    #Plot and save the results
    plt.figure(figsize=(16, 8), dpi=96)
    plt.subplot(1, 2, 1)
    plt.title('Hough Space')
    plt.imshow(hough[0])

    plt.subplot(1, 2, 2)
    plt.title('Circle Detection Result')
    plt.imshow(rgb_img)
    plt.axis('off')

    plt.savefig('output/hough_transform.png')
    plt.show()