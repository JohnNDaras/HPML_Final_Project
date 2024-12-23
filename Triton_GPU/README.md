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
This kernel computes the similarity metrics for all pairs of objects provided. It takes the following inputs:
- **Pointers**: Inputs to the kernel, such as areas, perimeters, bounding boxes, Fourier descriptors, etc.
- **Combination Pointers**: Information about pairs of objects to compare.
- **Tensor Sizes**: Block size and tensor size for the Triton kernel.

#### Arguments
- `keys_ptr`: Pointers to object keys.
- `areas_ptr`: Areas of the objects.
- `perimeters_ptr`: Perimeters of the objects.
- `bboxes_ptr`: Bounding box data.
- `fourier_ptr`: Fourier descriptors.
- `num_vertices_ptr`: Number of vertices for each object.
- `comb_keys_ptr`: Combinations of keys to compare.
- `intersection_areas_ptr`: Intersection areas for bounding boxes.
- `union_areas_ptr`: Union areas for bounding boxes.
- `n_combinations`: Total number of combinations to process.
- `BLOCK_SIZE`, `TENSOR_SIZE`: Configuration constants for the kernel.

### Helper Function: `compute_with_seventh_dir`
Manages preprocessing, kernel execution, and postprocessing.

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


