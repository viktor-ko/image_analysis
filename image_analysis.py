from basic_img_operations import image_io, spilt_and_merge, invert_color, image_statistics, histogram_exposure
from img_filters import padding_convolution, box_binomial_median_filters, sobel_filter, laplacian_filter
from morphology import erosion_dilation_opening_closing, road_extraction, car_road_fill, road_edge, \
    dist_transform_skeletonize, morph_result
from extraction_regions_edges import region_growing, watershed, mean_shift, slic_superpixels, douglas_peuke, \
    hough_transform

'''Image representations and basic operations'''
#Task 1: Load and visualize one color image in BGR color scale, the gray value image, and the image in RGB color scale
I, I_gray, I_RGB = image_io()

#Task 2: Spilt individual channels from one color image (i.e. R, G, B), and then merge them back to one single image in a different order (G, B, R)
r, g, b, I_merged = spilt_and_merge(I_RGB)

#Task 3: Point operation - apply the point operation and create a image with inverted color.
I_inverse = invert_color(I_RGB)

#Task 4: Statistical image characterization - calculate the mean and variance of the gray image and also for the RGB channels individually.
collect_mean, collect_variance = image_statistics(I_gray, r, g, b)

#Task 5: Calculate the histogram of gray scale images and observe their differences for over-exposure and under-exposure images.
histogram_exposure(I_RGB, I_gray)

'''Image filters'''
#Task 1: padding and convolution
padding_convolution()

#Task 2: Image smoothing - Box, Binomial and Median Filters
box_result, binomial_result, median_result = box_binomial_median_filters()

#Task 3: Sobel Filter
sobel_filter()

#Task 4: Laplacian Filter
laplacian_filter(binomial_result)

'''Morphological operations'''
#Task 1: Erosion and Dilation
erosion_dilation_opening_closing()

#Task 2: Road extraction
opening_img = road_extraction()

#Task 3: Get the car and fill the road
closing_img = car_road_fill(opening_img)

#Task 4: Get the edge of the road
edge_img = road_edge(closing_img)

#Task 5: Get the center line of the road
middle_line = dist_transform_skeletonize(closing_img)

#Show the final result
morph_result(edge_img, middle_line)

'''Extraction of edges and regions'''
#Task 1: Region Growing algorithm to segment the entire river part in the image
river_1 = region_growing()

#Task 2: Watershed Transformation to segment the river and both riverbanks in the image
watershed(river_1)

#Task 3: Mean Shift Segmentation
mean_shift()

#Task 4: Simple Linear Iterative Clustering (SLIC) superpixels algorithm
slic_superpixels()

#Task 5: Douglas Peuke Algorithm to approximates a polygonal curve: 'smooth' the boundary of the Leine River in Hannover
douglas_peuke()

#Task 6: Hough transformation to detect 4 small fountains in the Herrenhausen Gardens of Hannover
hough_transform()