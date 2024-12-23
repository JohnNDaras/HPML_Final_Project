# README

## Overview
This project implements a GPU-accelerated kernel using the Triton framework to compute similarity metrics between pairs of objects. It is designed for use in applications requiring geometric shape analysis, such as computer vision, object detection, and geospatial analysis. The code computes multiple similarity metrics and aggregates them into a final similarity score.

## Features
- **Circularity Similarity**: Measures how similar two objects are in terms of their circularity.
- **Perimeter Similarity**: Compares the perimeters of two objects.
- **Aspect Ratio Similarity**: Analyzes the aspect ratios of bounding boxes.
- **Bounding Box Distance**: Computes similarity based on the distance between the centers of bounding boxes.
- **Fourier Similarity**: Compares the Fourier descriptors of two objects.
- **Jacard Similarity**: Measures similarity based on the intersection over union (IoU).
- **Area Similarity**: Compares areas of the objects.
- **Curvature Similarity**: Evaluates the similarity in the number of vertices.
- **Final Combined Similarity**: Aggregates the above metrics into a weighted score.

## Requirements
- Python 3.8+
- PyTorch
- Triton
- Shapely
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install the required Python packages:
   ```bash
   pip install torch triton shapely numpy
   ```

## Usage

### Kernel Function: `compute_combinations_kernel`
This kernel computes the similarity metrics for all pairs of objects provided. It uses pointers and offsets for efficient memory access on the GPU.

#### Inputs and Memory Layout
- **Pointers**: The kernel accesses data stored in GPU memory using pointers. These pointers represent the starting addresses of data arrays in memory.
  - `keys_ptr`: Pointer to the object keys array.
  - `areas_ptr`: Pointer to the array storing areas of the objects.
  - `perimeters_ptr`: Pointer to the array storing perimeters of the objects.
  - `bboxes_ptr`: Pointer to the bounding box data array, where each bounding box is represented by 4 consecutive values.
  - `fourier_ptr`: Pointer to the array of Fourier descriptors, with each descriptor consisting of multiple consecutive values.
  - `num_vertices_ptr`: Pointer to the array storing the number of vertices for each object.
  - `comb_keys_ptr`: Pointer to the array of key combinations for pairwise comparison. Each combination consists of two indices (offsets) referring to the objects being compared.
  - `intersection_areas_ptr`: Pointer to the array storing intersection areas for bounding boxes.
  - `union_areas_ptr`: Pointer to the array storing union areas for bounding boxes.

#### Offsets
Offsets are used to access specific elements within these arrays:
- For combination pairs, the kernel retrieves two offsets from `comb_keys_ptr`, corresponding to the indices of the two objects being compared.
- For bounding boxes, offsets are multiplied by 4 to access the correct group of 4 consecutive values (representing the bounding box coordinates).
- For Fourier descriptors, offsets are multiplied by the tensor size (`TENSOR_SIZE`) to access the correct descriptor data.

#### Execution Flow
1. Compute the range of indices for the block of combinations being processed using the program ID (`pid`) and block size (`BLOCK_SIZE`).
2. Load the offsets for the current pair of objects using `comb_keys_ptr`.
3. Access the relevant data (areas, perimeters, bounding boxes, etc.) using these offsets.
4. Perform the similarity computations for each pair.
5. Store the results in the corresponding output arrays (`results1_ptr`, `results2_ptr`, etc.).

#### Configuration Constants
- `BLOCK_SIZE`: Determines the number of combinations processed by each block.
- `TENSOR_SIZE`: Specifies the number of elements in each Fourier descriptor.

### Class: `ShapeSimilarity`
This class provides tools for processing geometric shapes (polygons) and calculating similarities between clusters of shapes using precomputed properties.

#### Logic Overview
The `ShapeSimilarity` class stores all polygons in a dictionary where each polygon is uniquely identified by a key. The keys are the polygon indices. To organize polygons into groups, a list of sublists is used, where each sublist represents a cluster and contains the indices of polygons belonging to that cluster.

To manage memory usage efficiently for Triton processing:
- The dictionary is sorted in ascending order by its keys, and the keys are updated to be consecutive integers starting from 0.
- The sublists are also updated to reflect the new key mappings, ensuring they point to the same polygons as before.
- This approach is applied uniformly to all precomputed geometric properties (e.g., areas, perimeters, bounding boxes) to maintain consistency.

#### Key Methods

1. **`center_polygons`**
Centers polygons in a dictionary by translating each polygon so that its centroid is at the origin.

```python
Parameters:
    polygon_dict (dict): A dictionary where values are Shapely polygons.
Returns:
    dict: A dictionary with the same keys but with centered polygons as values.
```

2. **`sample_boundaries_vectorized`**
Samples evenly spaced points along the exterior boundary of multiple polygons.

```python
Parameters:
    polygons (array): Numpy array of Shapely polygons.
    num_points (int): Number of points to sample per polygon.
Returns:
    numpy.ndarray: A (num_polygons, num_points, 2) array of sampled points.
```

3. **`torch_fourier_descriptors`**
Computes Fourier Descriptors for a batch of contours using PyTorch.

```python
Parameters:
    contours (Tensor): Tensor of shape (num_contours, num_points, 2).
Returns:
    Tensor: Fourier descriptors of shape (num_contours, num_descriptors).
```

4. **`compute_fourier_descriptors`**
Precomputes Fourier descriptors for a set of polygons.

```python
Parameters:
    polygons (list): List of Shapely polygons.
    num_points (int): Number of points to sample along the boundaries.
Returns:
    Tensor: Fourier descriptors as a PyTorch tensor.
```

5. **`process_pairs_and_clusters_and_compute_areas`**
Generates unique pairs for each cluster and computes intersection and union areas for those pairs.

```python
Parameters:
    list_of_cluster_content_indices (list): List of cluster content indices.
    cluster_indices_list (list): List of cluster indices corresponding to each cluster.
Returns:
    tuple: Flattened pairs, group sizes, intersection areas, and union areas.
```

6. **`calculate_similarity_for_clusters`**
Calculates the similarity metrics for multiple clusters of polygons.

```python
Parameters:
    polygons (list): List of Shapely polygons.
    polygon_indices (list): Indices of polygons.
    list_of_cluster_content_indices (list): List of cluster indices.
    cluster_indices_list (list): List of cluster IDs.
Returns:
    dict: Average similarity scores for each cluster.
```

### Example

```python
from shapely.geometry import Polygon

# Define example polygons
polygons = [
    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),
    Polygon([(0, 0), (1, 0), (0.5, 1)])
]

# Define indices and clusters
polygon_indices = [10, 20, 30]  # Custom indices for the polygons
list_of_cluster_content_indices = [[10, 30], [20]]  # Two clusters
cluster_indices_list = [101, 200]

# Instantiate and calculate
shape_sim = ShapeSimilarity()
similarity_scores = shape_sim.calculate_similarity_for_clusters(
    polygons, polygon_indices, list_of_cluster_content_indices, cluster_indices_list
)

for cluster_id, similarity in similarity_scores.items():
    print(f"Cluster {cluster_id}: {similarity:.2f}%")
```

### Precomputed Properties
The following properties are precomputed for efficiency:
- **Areas**: Polygon areas.
- **Perimeters**: Polygon perimeters.
- **Bounding Boxes**: Bounds of polygons.
- **Fourier Descriptors**: Fourier coefficients for polygon shapes.
- **Intersection Areas and Union Areas**: For pairwise comparisons within clusters.

### Achievements with These Optimizations
The implemented optimizations achieve the following:
1. **Efficient Memory Usage**: By sorting and remapping the dictionary keys to consecutive integers, memory overhead is minimized, enabling smoother execution on GPUs.
2. **Consistency Across Properties**: Updating keys across all geometric properties ensures data alignment, reducing computational complexity and potential errors.
3. **Scalability**: These strategies allow handling large datasets and numerous clusters without exceeding memory limits, making the solution scalable for real-world applications.
4. **GPU Compatibility**: Precomputing and organizing properties like areas, perimeters, and Fourier descriptors streamline GPU processing, leveraging its parallel computation capabilities effectively.

## Performance Optimization
- **Batching**: The kernel processes data in blocks (specified by `BLOCK_SIZE`) for efficient parallelism.
- **Vectorized Operations**: Similarity metrics are computed using Triton's vectorized operations.

## Customization
- Modify weights in the final similarity score computation (`final_value`) to emphasize specific metrics.
- Adjust the `BLOCK_SIZE` and `TENSOR_SIZE` constants for optimal GPU performance based on hardware.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes or feature enhancements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


