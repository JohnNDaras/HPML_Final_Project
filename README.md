# High Performance Machine Learning Final Project

This repository contains the implementation of a high-performance machine learning framework for computing and analyzing spatial similarities in large geospatial datasets. The project is divided into three key directories, each focusing on a specific aspect of the workflow:

## Directory Overview

### 1. Datasets
This directory handles data extraction and preprocessing for geospatial datasets. It processes raw data from repositories like the US Census Bureau TIGER/Line Shapefiles and OpenStreetMap (OSM) to generate polygonal geometries for analysis.

**Key Features:**
- Batch processing and parallel data transformation.
- Configurable extraction limits for scalability.
- Outputs in CSV format for seamless integration with the rest of the workflow.

Refer to the [Datasets README](./Datasets/README.md) for more details.

---

### 2. Dynamic_Similarity_Algorithm
Contains the main algorithm for identifying high-similarity clusters efficiently. It leverages spatial indexing, neural networks, and kernel density estimation to achieve dynamic budget management and high recall accuracy.

**Key Features:**
- Spatial indexing for efficient geometry lookups.
- Machine learning-based cluster classification and verification.
- Multi-threaded operations for performance optimization.

Refer to the [Dynamic_Similarity_Algorithm README](./Dynamic_Similarity_Algorithm/README.md) for technical details and usage instructions.

---

### 3. Triton_GPU
Implements GPU-accelerated similarity computations using the Triton framework. The kernel calculates multiple similarity metrics—such as area, perimeter, and Fourier descriptors—across large datasets, optimizing performance for applications in computer vision and geospatial analysis.

**Key Features:**
- GPU-accelerated computation for similarity metrics.
- Support for large-scale datasets with efficient memory usage.
- Configurable metrics aggregation for tailored applications.

Refer to the [Triton_GPU README](./Triton_GPU/README.md) for more details.

---

## Features
- **Efficient Algorithms**: Combines spatial indexing, dynamic similarity evaluation, and machine learning for performance optimization.
- **Scalability**: Supports processing large geospatial datasets with configurable parameters and GPU acceleration.
- **Modular Design**: Clearly defined workflows in separate directories for ease of extension and integration.

## Usage
Navigate to the individual directories for detailed instructions and code examples:

- [Datasets](./Datasets/README.md)
- [Dynamic_Similarity_Algorithm](./Dynamic_Similarity_Algorithm/README.md)
- [Triton_GPU](./Triton_GPU/README.md)

---

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---
