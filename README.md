# Image Analysis for Mapping
This python scripts was created for the WS 2022/23 course "Image Analysis for Mapping" from the Cartography MSc. at the Technical University of Munich (TUM). It aims to explore different techniques for image analysis and processing to extract information from images and apply them to mapping tasks.

# Project Structure
- `image_analysis.py`: Main script that executes functions from modules below.
- `basic_img_operations.py`: Functions for basic image operations such as loading, visualizing, splitting, merging, and inverting images.
- `img_filters.py`: Functions for applying different image filters: box, binomial, median, Sobel, and Laplacian.
- `morphology.py`: Functions for morphological operations: erosion, dilation, opening, closing.
- `extraction_regions_edges.py`: Functions for extracting edges and regions using techniques like region growing, watershed transformation, mean shift segmentation, SLIC superpixels, Douglas-Peucker algorithm, and Hough transformation.
- `./images`: Directory containing sample images used in the project.
- `src_mat.npy`: Sample matrix used in morphological operations.

## Dependencies
- NumPy
- OpenCV
- Matplotlib
- Scikit-image

#  Covered Topics
## 1. Image representations and basic operations
### 1.1 Color scales

### 1.2 Spilt and merge image channels

### 1.3 Point operation - Inversion

### 1.4 Image Histograms


## 2. Image filters
### 2.1 Padding and Convolution

### 2.2 Image smoothing - Box, Binomial and Median Filters

### 2.3 Sobel Filter

### 2.4 Laplacian Filter


## 3. Morphological operations
### 3.1 Erosion, Dilation, Opening, Closing

### 3.2 Road extraction

### 3.3 Get the car and fill the road

### 3.4 Get the edge of the road 

### 3.5 Get the center line of the road (Distance Transformation and Skeletonization)

### 3.6 Final result


## 4. Extraction of edges and regions
### 4.1 Region Growing algorithm

### 4.2 Watershed Transformation

### 4.3 Mean Shift Segmentation

### 4.4 Simple Linear Iterative Clustering (SLIC) superpixels algorithm

### 4.5 Douglas Peuke Algorithm

### 4.6 Hough transformation