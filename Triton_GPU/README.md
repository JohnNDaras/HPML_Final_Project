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

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install the required Python packages:
   ```bash
   pip install torch triton
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

### Helper Function: `compute_with_seventh_dir`
Manages preprocessing, kernel execution, and postprocessing.

#### Preprocessing
- **Combination Generation**: Generates all pairwise combinations of object keys in the provided sublists. These combinations are flattened and stored in `all_combinations_flat`.
- **Intersection and Union Areas**: Flattens the provided intersection and union area data.

#### Kernel Execution
- **Memory Allocation**: Allocates GPU memory for input pointers and result arrays.
- **Grid Configuration**: Configures the grid size for kernel execution based on the total number of combinations and block size.
- **Kernel Call**: Passes the pointers, data, and configuration constants to the Triton kernel.

#### Postprocessing
- **Result Splitting**: Splits the flattened result arrays back into sublists corresponding to the original input sublists.

#### Example
```python
import torch

# Define inputs
keys = torch.tensor([...], device='cuda')
areas = torch.tensor([...], device='cuda')
perimeters = torch.tensor([...], device='cuda')
bboxes = torch.tensor([...], device='cuda')
fourier = torch.tensor([...], device='cuda')
num_vertices = torch.tensor([...], device='cuda')
key_lists = [[...], [...]]  # List of object keys
intersection_areas_dir = [[...], [...]]
union_areas_dir = [[...], [...]]

# Call helper function
results = compute_with_seventh_dir(
    keys, areas, perimeters, bboxes, fourier, num_vertices,
    key_lists, intersection_areas_dir, union_areas_dir
)

# Process results
for metric_results in results:
    print(metric_results)
```

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

