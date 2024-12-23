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

## Usage
```python
from dynamic_similarity import Dynamic_Similarity_Algorithm

algorithm = Dynamic_Similarity_Algorithm(
    budget=1000,
    delimiter=",",
    sourceFilePath="source.csv",
    targetFilePath="target.csv",
    testFilePath="clusters.csv",
    target_recall=0.9,
    similarity_index_range=0.1
)

algorithm.applyProcessing()
```

### Key Outputs
- **Timing Metrics:** Duration of indexing, preprocessing, training, and verification phases.
- **Recall Approximation:** Estimated recall based on the test set.
- **Verified Clusters:** True positives and relations between clusters.

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


