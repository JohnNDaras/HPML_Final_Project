# Dynamic Similarity Algorithm

## Overview
The `Dynamic_Similarity_Algorithm` is an advanced algorithm designed to efficiently compute similarity metrics between geometries across large datasets. The algorithm leverages a combination of geometric data processing, grid-based spatial indexing, and machine learning techniques (including KDE and neural networks) to identify clusters and evaluate their similarity while optimizing for performance and recall accuracy.

---

## Features
- **Spatial Indexing:** Efficiently indexes geometric data into grid cells to support fast candidate lookups.
- **Preprocessing:** Extracts, normalizes, and computes geometric and cluster-based features for training.
- **Model Training:** Uses a feedforward neural network to classify clusters.
- **Verification:** Applies trained models to verify cluster similarity and calculate true positives.
- **Kernel Density Estimation (KDE):** Estimates probability distributions for similarity thresholds.
- **Dynamic Budget Management:** Processes clusters dynamically while adhering to computational budgets.
- **Multi-threading:** Utilizes multiprocessing for parallel data loading and processing.

---

## Requirements

### Python Libraries
- Core Libraries: `math`, `numpy`, `random`, `sys`, `time`, `multiprocessing`, `os`, `collections`
- External Libraries: 
  - **Geometric Processing:** `shapely`
  - **Data Handling:** `pandas`, `sortedcontainers`
  - **Progress Tracking:** `tqdm`
  - **Machine Learning:** `tensorflow`, `scikit-learn`

### File Structure
- `utilities.py`: Contains helper classes such as `CsvReader` for data loading.
- `datamodel.py`: Defines data structures like `RelatedGeometries`.
- `shape_similarity.py`: Implements the `ShapeSimilarity` class for similarity calculations.
- `test.py`: Example script for running and testing the `Dynamic_Similarity_Algorithm`.

---

## Class Design
### `Dynamic_Similarity_Algorithm`
The main class encapsulates the complete workflow:

#### Attributes
- **Data Attributes:**
  - `sourceData`, `targetData`: Source and target geometries.
  - `cluster_data`: Cluster information including similarity indices.
  - `spatialIndex`: Grid-based spatial index.
  - `relations`: Stores verified cluster relations.
- **Algorithm Settings:**
  - `budget`: Maximum number of clusters to process.
  - `target_recall`: Desired recall rate.
  - `similarity_index_range`: Threshold range for similarity.
- **Feature Extraction Settings:**
  - `NO_OF_FEATURES`, `CLASS_SIZE`, `SAMPLE_SIZE`
- **Model Training Settings:**
  - `classifier`: Neural network model for classification.
  - `recall_aprox`, `detectedQP`: Metrics for recall approximation.

#### Methods
1. **Initialization:**
   ```python
   __init__(self, budget, delimiter, sourceFilePath, targetFilePath, testFilePath, target_recall, similarity_index_range)
   ```
   Loads data, sets thresholds, and initializes spatial grid parameters.

2. **Indexing:**
   ```python
   indexSource(self)
   addToIndex(self, geometryId, envelope)
   ```
   Indexes geometries into a spatial grid for fast lookup.

3. **Preprocessing:**
   ```python
   preprocessing(self)
   ```
   Extracts and normalizes features for model training and evaluation.

4. **Model Training:**
   ```python
   trainModel(self)
   ```
   Trains a neural network on geometric features using early stopping.

5. **Verification:**
   ```python
   verification(self)
   ```
   Uses the trained model to verify clusters and calculate true positives.

6. **Utility Functions:**
   - `getCandidates(self, targetId)`: Fetches candidates for a given geometry.
   - `get_feature_vector(self, sourceIds, targetIds, lengths)`: Constructs feature vectors.
   - `setThetas(self)`: Computes grid dimensions.
   - `validCandidate(self, candidateId, targetEnv)`: Validates candidate geometries.
   - KDE and Similarity Threshold:
     - `get_best_model(self, x_train, samples, h_vals, seed)`
     - `find_estimate_threshold(self, model, interpolation_points)`
     - `compute_estimate_cdf(self, model, target_range, interpolation_points)`

---

## Workflow

### Initialization
1. Load source, target, and cluster datasets using `CsvReader`.
2. Compute spatial grid dimensions (`thetaX`, `thetaY`).
3. Initialize spatial index and similarity relations.

### Processing Pipeline
1. **Indexing:**
   - Divide geometries into spatial grid cells.
   - Populate the grid with geometry IDs.

2. **Preprocessing:**
   - Extract geometric features (e.g., area, length, number of points).
   - Normalize features for training.
   - Identify and sample clusters for training and KDE.

3. **Model Training:**
   - Train a neural network using positive and negative clusters.
   - Use early stopping to prevent overfitting.

4. **Verification:**
   - Predict cluster probabilities with the trained model.
   - Verify relationships based on similarity thresholds.

---

## Visual Analysis
### Analogy of Checked to Targeted Clusters
The following visualizations illustrate the relationship between the percentage of targeted clusters and the corresponding percentage of checked clusters, across datasets D1, D2, and D3:

![Screenshot from 2024-12-21 03-15-13 (1)](https://github.com/user-attachments/assets/c64b4ea8-fdda-41ac-9ea6-ddfb0d5b5063)

#### Figure 4 Analysis
Figure 4 presents six detailed graphs that explore the dynamics between the top percentage of targeted clusters and the corresponding percentage of checked clusters, or their respective ratios, across three datasets (D1, D2, and D3). The rows in the figure correspond to specific datasets, highlighting the following:

1. **Top-Left (D1):**
   - Examines the analogy of checked clusters to targeted clusters for Dataset D1.
   - Shows how the percentage of checked clusters scales relative to the targeted clusters, reflecting the algorithm's efficiency in narrowing down high-similarity clusters.

2. **Top-Middle (D1):**
   - Displays the percentage of targeted clusters to checked clusters for Dataset D1.
   - Demonstrates the effort required to identify specific top clusters as the target percentage increases.

3. **Top-Right (D2):**
   - Analyzes the analogy of checked clusters to targeted clusters for Dataset D2.
   - Highlights similar trends as D1 but tailored to Dataset D2, emphasizing computational trade-offs required for identifying targeted clusters.

4. **Bottom-Left (D2):**
   - Visualizes the percentage of targeted clusters to checked clusters for Dataset D2.
   - Focuses on the trade-offs and efforts to achieve targeted clusters relative to the checked clusters.

5. **Bottom-Middle (D3):**
   - Depicts the analogy of checked clusters to targeted clusters for Dataset D3.
   - Explores how this ratio varies for larger clusters, showcasing the computational cost trends for large datasets.

6. **Bottom-Right (D3):**
   - Highlights the percentage of targeted clusters to checked clusters for Dataset D3.
   - Examines efficiency in identifying clusters, showing the impact of average cluster size on computational needs.

---

## Test Script: `test.py`
The `test.py` script provides a usage example for the `Dynamic_Similarity_Algorithm` class. It demonstrates how to set up and execute the algorithm with configurable parameters.

### Key Features
1. **Desired Recall Configuration:**
   - The script prompts the user to input the desired recall value. A high recall value (e.g., 0.9) is recommended for effective verification.

2. **Similarity Range Testing:**
   - Users can configure the similarity index range (e.g., 0.1, 0.3, 0.5) to analyze different ranges of similarity.

3. **Disregarding Budget:**
   - The budget is set to a very large value to ensure that all relevant clusters are processed, enabling comprehensive testing of similarity thresholds.

### Code Example
```python
from dynamic_similarity_algorithm import Dynamic_Similarity_Algorithm

main_dir = '../content/drive/MyDrive/hpml_final_project/D1/'

print('Enter desired recall:')
x = input()

sg = Dynamic_Similarity_Algorithm(
    budget=5631064,  # Large budget for comprehensive testing
    delimiter='\t',
    sourceFilePath=main_dir + "SourceDataset.csv",
    targetFilePath=main_dir + "TargetDataset.csv",
    testFilePath=main_dir + 'similarity_results.csv',
    target_recall=float(x),
    similarity_index_range=0.1  # Example similarity index range
)

sg.applyProcessing()
```

### Usage Notes
- Modify the `similarity_index_range` to experiment with different thresholds (e.g., 0.1, 0.3, 0.5).
- Ensure a high `target_recall` (e.g., 0.9) for better results in identifying clusters.
- The script uses a very large budget to prioritize recall over computational limits.

---

## Data Model: `datamodel.py`
### `RelatedGeometries`
The `RelatedGeometries` class in `datamodel.py` is responsible for managing and evaluating relationships between geometric clusters based on similarity thresholds.

#### Attributes
- **Cluster Statistics:**
  - `qualifyingClusters`: Total clusters qualifying for evaluation.
  - `similarity_threshold`: Threshold for determining similarity.
  - `verifiedClusters`: Number of clusters verified.
  - `interlinkedGeometries`: Number of geometries identified as related.
- **Progress Tracking:**
  - `pgr`: Progressive Geometry Recall metric.
  - `exceptions`, `violations`: Exception and violation counts during verification.
  - `continuous_unrelated_Clusters`: Counter for unrelated clusters.
- **Similarity Ranges:**
  - `similarity_0_10`, `similarity_10_20`, ..., `similarity_90_100`: Lists tracking clusters by similarity percentage ranges.

#### Methods
1. **Adding Similarities:**
   ```python
   addSimilarity(self, cluster, similarity)
   ```
   Categorizes clusters into similarity ranges based on their similarity scores.

2. **Retrieving Clusters:**
   ```python
   getNoOfClustersInRange(self, lower_bound, upper_bound)
   ```
   Returns the number of clusters within a specified similarity range.

3. **Resetting State:**
   ```python
   reset(self)
   ```
   Resets all attributes and clears similarity ranges.

4. **Verification:**
   ```python
   verifyRelations(self, cluster, similarity)
   ```
   Verifies if a cluster passes the similarity threshold and updates statistics accordingly.

5. **Output Results:**
   ```python
   print(self)
   ```
   Outputs the current state of the `RelatedGeometries` instance, including similarity distributions, recall, and precision metrics.

---

## Shape Similarity: `shape_similarity.py`
### `ShapeSimilarity`
The `ShapeSimilarity` class in `shape_similarity.py` provides a comprehensive set of methods to compute similarity metrics for geometric shapes, leveraging both spatial and mathematical properties.

#### Methods
1. **Centering Polygons:**
   ```python
   center_polygons(self, polygons)
   ```
   Translates polygons to have their centroids at the origin.

2. **Precomputing Properties:**
   ```python
   _precompute_properties(self, polygons)
   ```
   Precomputes and stores properties like area, perimeter, bounding boxes, and Fourier descriptors for efficient similarity computation.

3. **Fourier Descriptor:**
   ```python
   _fourier_descriptor(self, polygon, num_points=128)
   ```
   Computes Fourier descriptors for a polygon to capture its shape characteristics.

4. **Jaccard Similarity:**
   ```python
   jaccard_similarity(self, A, B)
   ```
   Computes the Jaccard similarity between two polygons based on their intersection and union areas.

5. **Area Similarity:**
   ```python
   area_similarity(self, idx_A, idx_B)
   ```
   Calculates similarity based on the overlapping area between two polygons.

6. **Curvature Similarity:**
   ```python
   curvature_similarity(self, idx_A, idx_B)
   ```
   Measures similarity based on the number of vertices in the polygons.

7. **Fourier Descriptor Similarity:**
   ```python
   fourier_descriptor_similarity(self, idx_A, idx_B)
   ```
   Computes similarity using Fourier descriptors.

8. **Aspect Ratio Similarity:**
   ```python
   aspect_ratio_similarity(self, bbox_A, bbox_B)
   ```
   Evaluates similarity based on the aspect ratios of bounding boxes.

9. **Perimeter Similarity:**
   ```python
   perimeter_similarity(self, idx_A, idx_B)
   ```
   Compares the perimeters of two polygons.

10. **Bounding Box Distance:**
    ```python
    bounding_box_distance(self, idx_A, idx_B)
    ```
    Calculates distance between the centers of bounding boxes.

11. **Polygon Circularity Similarity:**
    ```python
    polygon_circularity_similarity(self, idx_A, idx_B)
    ```
    Computes similarity based on the circularity of polygons.

12. **Combined Similarity:**
    ```python
    combined_similarity(self, idx_A, idx_B, ...)
    ```
    Aggregates multiple similarity measures into a weighted combined score.

13. **Calculate Similarity for All Pairs:**
    ```python
    calculate_similarity_all_pairs(self, polygons_array)
    ```
    Computes the average combined similarity score for all unique pairs of polygons in an array.

#### Usage Example
```python
from shape_similarity import ShapeSimilarity
from shapely.geometry import Polygon

# Create polygons
polygon1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
polygon2 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

# Initialize ShapeSimilarity
similarity_calculator = ShapeSimilarity()

# Calculate Jaccard similarity
jaccard = similarity_calculator.jaccard_similarity(polygon1, polygon2)
print(f"Jaccard Similarity: {jaccard}")

# Calculate combined similarity
similarity_score = similarity_calculator.calculate_similarity_all_pairs([polygon1, polygon2])
print(f"Combined Similarity Score: {similarity_score}")
```

---

## Utilities: `utilities.py`
### `CsvReader`
The `CsvReader` class in `utilities.py` provides utility methods for reading and processing CSV files containing geometric data.

#### Methods
1. **Read All Entities:**
   ```python
   readAllEntities(delimiter, inputFilePath, batch_size=1000)
   ```
   Reads geometric data from a CSV file, processes them in batches, and converts WKT representations into Shapely geometry objects.
   - **Args:**
     - `delimiter` (str): Delimiter used in the CSV file.
     - `inputFilePath` (str): Path to the CSV file.
     - `batch_size` (int, optional): Number of rows to process in each batch. Default is 1000.
   - **Returns:**
     - List of Shapely geometry objects.

2. **Load Cluster Data to Deque:**
   ```python
   loadClusterDataToDeque(inputFilePath)
   ```
   Loads cluster data from a CSV file into a deque for efficient appending and popping.
   - **Assumes:** The CSV file contains two columns:
     - `cluster_id` (int)
     - `similarity_index` (float)
   - **Args:**
     - `inputFilePath` (str): Path to the CSV file.
   - **Returns:**
     - A deque of tuples containing cluster IDs and their similarity indices.

#### Usage Example
```python
from utilities import CsvReader

# Load geometric data from a CSV file
geometries = CsvReader.readAllEntities(delimiter=",", inputFilePath="data.csv")

# Load cluster data into a deque
cluster_data = CsvReader.loadClusterDataToDeque(inputFilePath="clusters.csv")
print(cluster_data)
```

---

## Performance Optimization
- **Parallel Processing:** Utilizes multiprocessing for data loading.
- **Grid Indexing:** Reduces computational complexity by spatial indexing.
- **Dynamic Budgeting:** Adapts the number of verified clusters to the similarity range and recall targets.
- **Sorted Containers:** The algorithm uses `SortedList` from the `sortedcontainers` library to efficiently sort cluster predictions by probability. This ensures that only the top clusters, within a defined `maxsize`, are verified. By limiting verification to a portion of the dataset, the algorithm achieves computational efficiency while maintaining a recall close to the desired target.

---

## Future Work
- Improve grid-based indexing for non-uniform distributions.
- Extend model training to include additional features or advanced architectures.
- Optimize verification for higher scalability in massive datasets.

---
