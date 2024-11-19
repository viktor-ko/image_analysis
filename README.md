# Image Analysis for Mapping
These Python scripts were created for the WS 2022/23 "Image Analysis for Mapping" course from the Cartography MSc. at the Technical University of Munich (TUM). They demonstrate different image analysis and processing techniques to extract information from images and use it in common mapping tasks.

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
![bgr_rgb_gray](https://github.com/user-attachments/assets/bf61040c-d20f-4391-b5d6-dd87041d9a86)

### 1.2 Spilt and merge image channels
![r_g_b_merge](https://github.com/user-attachments/assets/24d4327b-54dc-4edb-8f58-3c6d4698d211)

### 1.3 Point operation - Inversion
![inverted](https://github.com/user-attachments/assets/8a6b3fcb-41bc-4e6b-9ceb-eb80343c0ce0)

### 1.4 Image Histograms
![normal_over_under](https://github.com/user-attachments/assets/cc6bfe52-2dc1-4142-833a-2bdb671697f6)
![histogram](https://github.com/user-attachments/assets/dbb76b32-01c3-4205-afdd-d61b4aa75bbf)
![histogram_color](https://github.com/user-attachments/assets/174453ff-126e-460c-815e-37f821776640)

## 2. Image filters
### 2.1 Padding and Convolution
![padding_convolution](https://github.com/user-attachments/assets/4d593d00-a894-4510-9e79-41dfd93c25ac)

### 2.2 Image smoothing - Box, Binomial and Median Filters
![box_binomial_median_filters](https://github.com/user-attachments/assets/51de7a79-c861-4300-a11d-e2ee85002b3d)

### 2.3 Sobel Filter
![sobel_filter](https://github.com/user-attachments/assets/e6a4e566-a40f-42e8-8756-caba93c2103d)

### 2.4 Laplacian Filter
![laplacian_filter](https://github.com/user-attachments/assets/fec60d29-09da-41f2-9769-f47bebe9537a)

## 3. Morphological operations
### 3.1 Erosion, Dilation, Opening, Closing
![erosion_dilation_opening_closing](https://github.com/user-attachments/assets/a2d5516f-509a-4c42-9977-c8fa36ac8340)

### 3.2 Road extraction
![road_extraction](https://github.com/user-attachments/assets/4879f7f2-4fb5-4207-b836-cc3114308425)

### 3.3 Get the car and fill the road
![car_road_fill](https://github.com/user-attachments/assets/f97d0ad6-644b-45a8-a6af-e06b9407da56)

### 3.4 Get the edge of the road 
![road_edge](https://github.com/user-attachments/assets/30f481a4-25a4-4c0a-bd83-0ffd3f4126fc)

### 3.5 Get the center line of the road (Distance Transformation and Skeletonization)
![road_center](https://github.com/user-attachments/assets/e97ea005-78d9-4d19-ba7a-b1fce13fcbb2)

### 3.6 Final result
![morph_result](https://github.com/user-attachments/assets/7bf77073-e37c-447f-91ff-60aef8e4adc3)


## 4. Extraction of edges and regions
### 4.1 Region Growing algorithm
![region_growing](https://github.com/user-attachments/assets/71d149fa-6f13-4c86-8db7-e7fbdb8352c7)

### 4.2 Watershed Transformation
![watershed](https://github.com/user-attachments/assets/b5c78aa7-dda6-44c1-853d-75c250cfb525)

### 4.3 Mean Shift Segmentation
![mean_shift](https://github.com/user-attachments/assets/0efa5116-b123-4fdd-bc40-cc944e83c3d6)

### 4.4 Simple Linear Iterative Clustering (SLIC) superpixels algorithm
![slic_superpixels](https://github.com/user-attachments/assets/585ff6b6-de47-42ce-86b0-ccc4c5734314)

### 4.5 Douglas Peuke Algorithm
![douglas_peuke](https://github.com/user-attachments/assets/70df8e8d-e7bf-4b52-8747-f877fdea89d2)

### 4.6 Hough transformation
![hough_transform](https://github.com/user-attachments/assets/fe6e0854-5096-419b-a3dd-5e0c1ca830bc)
