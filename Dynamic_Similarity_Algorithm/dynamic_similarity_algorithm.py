import math
import numpy as np
import random
import sys
import time
import shapely
import pandas as pd
from collections import defaultdict
import os
import random
from tqdm import tqdm
import multiprocessing
from sortedcontainers import SortedList

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from shapely import get_num_coordinates
from utilities import CsvReader
from datamodel import RelatedGeometries
from shape_similarity import ShapeSimilarity


class Dynamic_Similarity_Algorithm:

    def __init__(self, budget: int, qClusters: int, avSimilarity: float, delimiter: str, sourceFilePath: str, targetFilePath: str, testFilePath: str, target_recall: float, similarity_index_range: float):
        self.CLASS_SIZE = 500
        self.NO_OF_FEATURES = 16
        self.SAMPLE_SIZE = 50000
        self.budget = budget
        self.target_recall = target_recall
        self.similarity_index_range = similarity_index_range
        self.detectedQP = 0
        self.recall_aprox = 0
        self.trainingPhase = False
        self.thetaX, self.thetaY = 0, 0


        # Load and process datasets using multiprocessing
        with multiprocessing.Pool(processes=2) as pool:
            results = pool.starmap(CsvReader.readAllEntities, [
                (delimiter, sourceFilePath),
                (delimiter, targetFilePath)
            ])
        
        # Store processed source and target data
        self.sourceData, self.targetData = results

        print('Source geometries:', len(self.sourceData))
        print('Target geometries:', len(self.targetData))

        # Load cluster data into a deque
        self.cluster_data = CsvReader.loadClusterDataToDeque(testFilePath)
        print('Cluster data:', len(self.cluster_data))

        # Convert deque to dictionary for faster lookups
        self.cluster_data = {cluster_id: similarity_index for cluster_id, similarity_index in self.cluster_data}

        self.kde_sample = set()
        self.predicted_probabilities = []
        self.relations = RelatedGeometries(qClusters, avSimilarity)
        self.similarity_calculator = ShapeSimilarity()
        self.sample = set()
        self.spatialIndex = defaultdict(lambda: defaultdict(list))
        self.verifiedClusters = set()

    def applyProcessing(self) :
      """
      Execute the complete algorithm pipeline:
      1. Index the source geometries.
      2. Perform preprocessing to generate feature vectors.
      3. Train the neural network model.
      4. Verify clusters using the trained model.
      """
      time1 = int(time.time() * 1000)
      # Index geometries from the source data, for cluster extraction
      self.setThetas()
      self.indexSource()
      time2 = int(time.time() * 1000)
      # Generate sample of clusters for training and kde. Preprocess feature vectors for normalization
      self.preprocessing()
      time3 = int(time.time() * 1000)
      # Train the neural network model
      self.trainModel()
      time4 = int(time.time() * 1000)
      # Perform verification to select the best Clusters
      self.verification()
      time5 = int(time.time() * 1000)
      # Print timing information for each phase
      print("Indexing Time\t:\t" + str(time2 - time1))
      print("Initialization Time\t:\t" + str(time3 - time2))
      print("Training Time\t:\t" + str(time4 - time3))
      print("Verification Time\t:\t" + str(time5 - time4))
      self.relations.print()

    def indexSource(self) :
      """
      Index source geometries into a spatial grid for fast candidate lookup.
      Each geometry's bounding box is divided into grid cells.
      """
      geometryId = 0
      for sEntity in self.sourceData:
        self.addToIndex(geometryId, sEntity.bounds)
        geometryId += 1

    def addToIndex(self, geometryId, envelope) :
        """
        Add a geometry to the spatial grid index.

        Args:
            geometryId (int): The ID of the geometry.
            envelope (tuple): The bounding box of the geometry.
        """
        # Calculate the grid cell indices for the bounding box
        maxX = math.ceil(envelope[2] / self.thetaX)
        maxY = math.ceil(envelope[3] / self.thetaY)
        minX = math.floor(envelope[0] / self.thetaX)
        minY = math.floor(envelope[1] / self.thetaY)
        # Add the geometry ID to each relevant grid cell
        for latIndex in range(minX, maxX+1):
          for longIndex in range(minY, maxY+1):
              self.spatialIndex[latIndex][longIndex].append(geometryId)


    def preprocessing(self):
        """
        Preprocess the data to prepare for model training and evaluation:
        - Samples candidate clusters for training and KDE-based similarity computation.
        - Calculates geometric features and normalizes them.
        - Identifies clusters for more efficient feature computation.
        """
        sampled_ids = set()
        kde_sample_ids = set()
        max_candidates = 10 * len(self.sourceData)

        sampled_ids = set(random.sample(range(max_candidates + 1), self.SAMPLE_SIZE)) #sample for training

        # Create the remaining pool of candidates by excluding the first sample
        remaining_candidates = set(range(max_candidates + 1)) - sampled_ids

        # Generate the second set of random IDs from the remaining pool
        kde_sample_ids = set(random.sample(remaining_candidates, min(1000, len(remaining_candidates)))) #kde sample

        # Ensure the two sets are disjoint
        assert sampled_ids.isdisjoint(kde_sample_ids), "Sets should have no common elements!"

        # Initialize feature-related arrays for source data
        self.flag = [-1] * len(self.sourceData)
        self.frequency = [-1] * len(self.sourceData)
        self.distinctCooccurrences =  [0] * len(self.sourceData)
        self.realCandidates =  [0] * len(self.sourceData)
        self.totalCooccurrences =  [0] * len(self.sourceData)
        self.maxFeatures = [-sys.float_info.max] * self.NO_OF_FEATURES
        self.minFeatures = [sys.float_info.max] * self.NO_OF_FEATURES
        self.maxClusterFeatures = [-sys.float_info.max] * self.NO_OF_FEATURES
        self.minClusterFeatures = [sys.float_info.max] * self.NO_OF_FEATURES

        # Initialize candidate clusters data structures
        self.totalCandidateClusters = 0
        self.sourceDataCandidates_indexes = set()
        sourceDataCandidates_indexes_pairings = []
        targetData_indexes_pairings = []
        self.sourceDataCandidates = []
        lengths = []

        # Process target data to identify candidate matches (clusters)
        targetGeomId, ClusterId = 0, 0
        for targetId in range(len(self.targetData)):
          # Retrieve candidate matches for the current target. The candidate matches of a targetId represent a cluster.
          candidateMatches = self.getCandidates(targetId)
          currentCandidates = 0
          currentDistinctCooccurrences = len(candidateMatches)
          currentCooccurrences = 0
          if len(candidateMatches) > 1:
              # Add targetId (cluster's representative geometry) to the sample or KDE sample set if applicable
              if (ClusterId in sampled_ids):
                self.sample.add(targetId)

              if (ClusterId in kde_sample_ids):
                self.kde_sample.add(targetId)

              for candidateMatchId in candidateMatches:
                  # Append Cluster lengths
                  lengths.append(len(candidateMatches))
                  self.sourceDataCandidates_indexes.add(candidateMatchId)
                  sourceDataCandidates_indexes_pairings.append(self.sourceData[candidateMatchId])
                  targetData_indexes_pairings.append(self.targetData[targetId])

                  # Update statistics for candidate matching we use in feature extraction
                  self.distinctCooccurrences[candidateMatchId] += 1
                  currentCooccurrences += self.frequency[candidateMatchId]
                  self.totalCooccurrences[candidateMatchId] += self.frequency[candidateMatchId]

                  # Check if the candidate is valid and update counts
                  if self.validCandidate(candidateMatchId, self.targetData[targetId].envelope):
                      currentCandidates += 1
                      self.realCandidates[candidateMatchId] += 1

              # Update feature min and max values of individual geometries across all dataset
              self.maxFeatures[12] = max(self.maxFeatures[12], currentCooccurrences)
              self.minFeatures[12] = min(self.minFeatures[12], currentCooccurrences)

              self.maxFeatures[13] = max(self.maxFeatures[13], currentDistinctCooccurrences)
              self.minFeatures[13] = min(self.minFeatures[13], currentDistinctCooccurrences)

              self.maxFeatures[14] = max(self.maxFeatures[14], currentCandidates)
              self.minFeatures[14] = min(self.minFeatures[14], currentCandidates)
          ClusterId += 1

        # Generate a subset of source data. These are the geometries of all clusters
        self.sourceDataCandidates =[self.sourceData[idx] for idx in self.sourceDataCandidates_indexes]

        # Compute geometric properties of candidate pairings (source geometry - target geometry)

        #Compute Envelopes
        SourceGeomEnvelopes = shapely.envelope(sourceDataCandidates_indexes_pairings)
        TargetGeomEnvelopes = shapely.envelope(targetData_indexes_pairings)

        #Compute lengths
        SourceGeomLength = shapely.length(sourceDataCandidates_indexes_pairings)
        TargetGeomLength = shapely.length(targetData_indexes_pairings)

        #Compute Bounds
        sourceBounds = shapely.bounds(sourceDataCandidates_indexes_pairings)
        targetBounds = shapely.bounds(targetData_indexes_pairings)
        SourceBlocks = self.getNoOfBlocks1(sourceBounds)
        TargetBlocks = self.getNoOfBlocks1(targetBounds)

        #Compute Number of Coords
        source_no_of_points = self.getNoOfPoints(sourceDataCandidates_indexes_pairings)
        target_no_of_points = self.getNoOfPoints(targetData_indexes_pairings)

        #Compute envelope areas
        sourceDataAreas = shapely.area(SourceGeomEnvelopes)
        targetDataAreas = shapely.area(TargetGeomEnvelopes)

        #Find MBR intersection
        pairs = list(zip(SourceGeomEnvelopes, TargetGeomEnvelopes))
        MbrIntersection = shapely.area(shapely.intersection(*np.transpose(pairs)))

        self.maxFeatures[0] = max(sourceDataAreas)
        self.minFeatures[0] = min(sourceDataAreas)

        self.maxFeatures[1] = max(targetDataAreas)
        self.minFeatures[1] = min(targetDataAreas)

        self.maxFeatures[2] = max(SourceBlocks)
        self.minFeatures[2] = min(SourceBlocks)

        self.maxFeatures[3] = max(TargetBlocks)
        self.minFeatures[3] = min(TargetBlocks)

        self.maxFeatures[4] = max(self.frequency)
        self.minFeatures[4] = min(self.frequency)

        self.maxFeatures[5] = max(source_no_of_points)
        self.minFeatures[5] = min(source_no_of_points)

        self.maxFeatures[6] = max(target_no_of_points)
        self.minFeatures[6] = min(target_no_of_points)

        self.maxFeatures[7] = max(SourceGeomLength)
        self.minFeatures[7] = min(SourceGeomLength)

        self.maxFeatures[8] = max(TargetGeomLength)
        self.minFeatures[8] = min(TargetGeomLength)

        self.maxFeatures[9] = max(self.totalCooccurrences)
        self.minFeatures[9] = min(self.totalCooccurrences)

        self.maxFeatures[10] = max(self.distinctCooccurrences)
        self.minFeatures[10] = min(self.distinctCooccurrences)

        self.maxFeatures[11] = max(self.realCandidates)
        self.minFeatures[11] = min(self.realCandidates)


        # Calculate the mean values of the above geometric properties for each cluster
        cumsum_lengths = np.cumsum([0] + lengths)

        # Use list comprehension to calculate means

        SourceClusterDataAreas = [np.mean(sourceDataAreas[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]
        TargetClusterDataAreas = [np.mean(targetDataAreas[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]

        source_Cluster_no_of_points = [np.mean(source_no_of_points[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]
        target_Cluster_no_of_points = [np.mean(target_no_of_points[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]

        SourceClusterBlocks = [np.mean(SourceBlocks[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]
        TargetClusterBlocks = [np.mean(TargetBlocks[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]

        SourceClusterGeomLength = [np.mean(SourceGeomLength[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]
        TargetClusterGeomLength = [np.mean(TargetGeomLength[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]

        MbrClusterIntersection = [np.mean(MbrIntersection[start:end]) for start, end in zip(cumsum_lengths[:-1], cumsum_lengths[1:])]

        # Find the min and max "mean values" for each cluster
        self.maxClusterFeatures[0] = max(SourceClusterDataAreas)
        self.minClusterFeatures[0] = min(SourceClusterDataAreas)

        self.maxClusterFeatures[1] = max(TargetClusterDataAreas)
        self.minClusterFeatures[1] = min(TargetClusterDataAreas)

        self.maxClusterFeatures[2] = max(SourceClusterBlocks)
        self.minClusterFeatures[2] = min(SourceClusterBlocks)

        self.maxClusterFeatures[3] = max(TargetClusterBlocks)
        self.minClusterFeatures[3] = min(TargetClusterBlocks)

        self.maxClusterFeatures[4] = max(self.frequency)
        self.minClusterFeatures[4] = min(self.frequency)

        self.maxClusterFeatures[5] = max(source_Cluster_no_of_points)
        self.minClusterFeatures[5] = min(source_Cluster_no_of_points)

        self.maxClusterFeatures[6] = max(target_Cluster_no_of_points)
        self.minClusterFeatures[6] = min(target_Cluster_no_of_points)

        self.maxClusterFeatures[7] = max(SourceClusterGeomLength)
        self.minClusterFeatures[7] = min(SourceClusterGeomLength)

        self.maxClusterFeatures[8] = max(TargetClusterGeomLength)
        self.minClusterFeatures[8] = min(TargetClusterGeomLength)

        self.maxClusterFeatures[9] = self.maxFeatures[9]
        self.minClusterFeatures[9] = self.minFeatures[9]

        self.maxClusterFeatures[10] = self.maxFeatures[10]
        self.minClusterFeatures[10] = self.minFeatures[10]

        self.maxClusterFeatures[11] = self.maxFeatures[11]
        self.minClusterFeatures[11] = self.minFeatures[11]

        self.maxClusterFeatures[12] = self.maxFeatures[12]
        self.minClusterFeatures[12] = self.minFeatures[12]

        self.maxClusterFeatures[13] = self.maxFeatures[13]
        self.minClusterFeatures[13] = self.minFeatures[13]

        self.maxClusterFeatures[14] = self.maxFeatures[14]
        self.minClusterFeatures[14] = self.minFeatures[14]


    def getNoOfPoints(self, geometries):
        # Using vectorized get_num_coordinates to count points in each geometry
        return get_num_coordinates(geometries)


    def getCandidates(self, targetId):
        candidates = set()

        targetGeom = self.targetData[targetId]
        envelope = targetGeom.envelope.bounds
        maxX = math.ceil(envelope[2] / self.thetaX)
        maxY = math.ceil(envelope[3] / self.thetaY)
        minX = math.floor(envelope[0] / self.thetaX)
        minY = math.floor(envelope[1] / self.thetaY)

        for latIndex in range(minX, maxX+1):
          for longIndex in range(minY,maxY+1):
              for sourceId in self.spatialIndex[latIndex][longIndex]:
                  if (self.flag[sourceId] != targetId): 
                      self.flag[sourceId] = targetId
                      self.frequency[sourceId] = 0
                  self.frequency[sourceId] += 1
                  candidates.add(sourceId)

        return candidates

    def setThetas(self):
        """
        Compute average grid cell dimensions (thetaX, thetaY) based on source geometries.
        This determines the size of each spatial grid cell.
        """
        self.thetaX, self.thetaY = 0, 0
        for sEntity in self.sourceData:
            envelope = sEntity.envelope.bounds
            self.thetaX += envelope[2] - envelope[0]
            self.thetaY += envelope[3] - envelope[1]

        self.thetaX /= len(self.sourceData)
        self.thetaY /= len(self.sourceData)
        print("Dimensions of Equigrid", self.thetaX,"and", self.thetaY)

    def validCandidate(self, candidateId, targetEnv):
        return self.sourceData[candidateId].envelope.intersects(targetEnv)

    @staticmethod
    def create_model(input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        return model



    def trainModel(self):
      """
      Train a feedforward neural network model to classify candidate clusters.
      - Samples positive and negative clusters based on the KDE similarity threshold.
      - Constructs feature vectors for training.
      - Uses early stopping to avoid overfitting.
      """
      self.trainingPhase = True
      self.sample = list(self.sample)
      random.shuffle(self.sample)
      negativeClassFull, positiveClassFull = False, False
      SourceIds, TargetIds = [], []
      TestSourceIds, TestTargetIds = [], []
      excesive_positive_sourceIds, excesive_positive_targetIds = [], []
      excesive_positive_verifications = set()
      lengths = []
      test_lengths = []
      X, y = [], []
      sorted_list = SortedList()
      average_similarities = []
      excessVerifications = 0
      test_positives = 0
      epsilon = - self.similarity_index_range/4

      #Use KDE sample to find similarity threshold
      for targetId in self.kde_sample:
          candidateMatches = self.getCandidates(targetId)
          if len(candidateMatches) == 0:
              continue
          test_lengths.append(len(candidateMatches))
          [TestSourceIds.append(candidate) for candidate in candidateMatches]
          [TestTargetIds.append(targetId) for i in range(len(candidateMatches))]

          # Map candidateMatches indexes to their corresponding polygons
          candidatePolygons = [self.sourceData[idx] for idx in candidateMatches]
          average_similarity = self.similarity_calculator.calculate_similarity_all_pairs(candidatePolygons)
          average_similarities.append(average_similarity/100)

      #Find similarity threshold
      df_average_similarities = pd.DataFrame({'0': average_similarities})
      df_average_similarities = df_average_similarities['0']
      kde_model2 = self.get_best_model(df_average_similarities)
      self.find_estimate_threshold(kde_model2)
      print("Threshold", self.similarity_index_threshold)

      # Train the model using similarity threshold found with KDE
      for targetId in self.sample:
          if negativeClassFull and positiveClassFull:
              break
          candidateMatches = self.getCandidates(targetId) #find cluster for each targetId in training sample
          if len(candidateMatches) == 0:
              continue

          # Map candidateMatches indexes to their corresponding polygons
          candidatePolygons = [self.sourceData[idx] for idx in candidateMatches]
          average_similarity = self.similarity_calculator.calculate_similarity_all_pairs(candidatePolygons)

          if average_similarity >= self.similarity_index_threshold*100:
                  if y.count(1) < self.CLASS_SIZE:
                      y.append(1)
                      lengths.append(len(candidateMatches))
                      [SourceIds.append(candidate) for candidate in candidateMatches]
                      [TargetIds.append(targetId) for i in range(len(candidateMatches))]
                  else:
                      excessVerifications += 1
                      positiveClassFull = True
          else:
                  if y.count(0) < self.CLASS_SIZE:
                      y.append(0)
                      lengths.append(len(candidateMatches))
                      [SourceIds.append(candidate) for candidate in candidateMatches]
                      [TargetIds.append(targetId) for i in range(len(candidateMatches))]
                  else:
                      excessVerifications += 1
                      negativeClassFull = True

      self.N = y.count(1) + y.count(0) + excessVerifications
      total = sum(lengths)

      X = self.get_feature_vector(SourceIds, TargetIds, lengths)

      # Ensure both positive and negative classes are represented
      if y.count(0) == 0 or y.count(1) == 0:
          raise ValueError("Both negative and positive instances must be labelled.")

      y = np.array(y)

      # Validate X
      if not isinstance(X, np.ndarray) or len(X.shape) != 2:
          raise ValueError("X must be a 2D NumPy array, but got: {}".format(X))

      # Create and compile the neural network model
      model = Dynamic_Similarity_Algorithm.create_model(X.shape[1])
      model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

      # Train the model
      model.fit(X, y, epochs=30, batch_size=32, validation_split=0.1, verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

      self.classifier = model  # Store the trained model

      #SIMULATE VERIFICATION PHASE TO TO FIND APPROXIMATE RECALL
      Instances = self.get_feature_vector(TestSourceIds, TestTargetIds, test_lengths)
      predictions = self.classifier.predict(Instances)

      for pred, idx in tqdm(zip(predictions, average_similarities), desc="Calculating Recall", unit="prediction"):
          weight = float(pred[0])
          sorted_list.add((weight, idx))

      # Reverse and convert to a list
      reversed_list = list(reversed(sorted_list))
      
      #Find how many similarities predicted to be above similarity threshold in a range = similarity_range * len(reversed_list)
      test_positives = sum(1 for _, idx in reversed_list[:int(len(reversed_list) * (self.similarity_index_range + epsilon))] if idx >= self.similarity_index_threshold)
      
      #Find how many similarities are actually above similarity threshold in a range = similarity_range * len(reversed_list)
      count_above_similarity_threshold = len([x for x in average_similarities if x >= self.similarity_index_threshold])
      print("count_above_similarity_threshold = ", count_above_similarity_threshold)

      self.detectedQP = count_above_similarity_threshold
      self.N = len(self.kde_sample)

      #Estimate Recall Approximation
      self.recall_aprox = test_positives / count_above_similarity_threshold

      print("int(len(sorted_list) // (1/self.similarity_index_range - epsilon)) = ", int(len(reversed_list) * (self.similarity_index_range + epsilon)))
      print("test_positives = ", test_positives)
      print("count_above_similarity_threshold = ", count_above_similarity_threshold)
      print("len(self.kde_sample) = ", len(self.kde_sample))
      print("recall_aprox = ", self.recall_aprox)

      self.trainingPhase = False


    def get_feature_vector(self, sourceIds, targetIds, lengths):
        # Create a matrix to hold the feature vectors
        featureVectors = np.zeros((len(sourceIds), self.NO_OF_FEATURES))

        # Extract geometries based on source and target IDs
        sourceGeometries = np.array([self.sourceData[sourceId] for sourceId in sourceIds])
        targetGeometries = np.array([self.targetData[targetId] for targetId in targetIds])
        
        '''
         Calculate geometries' envelopes, bounds, intersection areas, etc. utilizing vectorizing operations of Shapely 2.0
         for each pair of souce and target ID .
        '''
        SourceGeomEnvelopes = shapely.envelope(sourceGeometries)
        TargetGeomEnvelopes = shapely.envelope(targetGeometries)

        SourceGeomEnvelopesBounds = shapely.bounds(SourceGeomEnvelopes)
        TargetGeomEnvelopesBounds = shapely.bounds(TargetGeomEnvelopes)

        pairs = list(zip(SourceGeomEnvelopes, TargetGeomEnvelopes))
        mbrIntersection = shapely.area(shapely.intersection(*np.transpose(pairs)))

        SourceGeomAreas = shapely.area(SourceGeomEnvelopes)
        TargetGeomAreas = shapely.area(TargetGeomEnvelopes)

        SourceGeomLenght = shapely.length(sourceGeometries)
        TargetGeomLenght = shapely.length(targetGeometries)

        sourceBounds = shapely.bounds(sourceGeometries)
        targetBounds = shapely.bounds(targetGeometries)

        SourceBlocks = self.getNoOfBlocks1(sourceBounds)
        TargetBlocks = self.getNoOfBlocks1(targetBounds)

        source_no_of_points = self.getNoOfPoints(sourceGeometries)
        target_no_of_points = self.getNoOfPoints(targetGeometries)

        # Populate the feature vector matrix
        for i, (sourceGeom, targetGeom) in enumerate(zip(sourceIds, targetIds)):
            candidateMatches = self.getCandidates(targetGeom)
            featureVectors[i, 12] = sum(self.frequency[candidateMatchId] for candidateMatchId in candidateMatches)
            featureVectors[i, 13] = len(candidateMatches)

            if mbrIntersection[i] > 0:
                featureVectors[i, 14] += 1

            # Area-based features
            featureVectors[i, 0] = (SourceGeomAreas[i]  - self.minFeatures[0]) / self.maxFeatures[0] * 10000
            featureVectors[i, 1] = (TargetGeomAreas[i] - self.minFeatures[1]) / self.maxFeatures[1] * 10000
            #featureVectors[i, 2] = (mbrIntersection[i] - self.minFeatures[2]) / self.maxFeatures[2] * 10000

            # Grid-based features
            featureVectors[i, 2] = (SourceBlocks[i] - self.minFeatures[2]) / self.maxFeatures[2] * 10000
            featureVectors[i, 3] = (TargetBlocks[i] - self.minFeatures[3]) / self.maxFeatures[3] * 10000
            featureVectors[i, 4] = (self.frequency[sourceIds[i]] - self.minFeatures[4]) / self.maxFeatures[4] * 10000

            # Boundary-based features
            featureVectors[i, 5] = (source_no_of_points[i] - self.minFeatures[5]) / self.maxFeatures[5] * 10000
            featureVectors[i, 6] = (target_no_of_points[i] - self.minFeatures[6]) / self.maxFeatures[6] * 10000
            featureVectors[i, 7] = (SourceGeomLenght[i] - self.minFeatures[7]) / self.maxFeatures[7] * 10000
            featureVectors[i, 8] = (TargetGeomLenght[i] - self.minFeatures[8]) / self.maxFeatures[8] * 10000

            # Candidate-based features
            featureVectors[i, 9] = (self.totalCooccurrences[sourceIds[i]] - self.minFeatures[9]) / self.maxFeatures[9] * 10000
            featureVectors[i, 10] = (self.distinctCooccurrences[sourceIds[i]] - self.minFeatures[10]) / self.maxFeatures[10] * 10000
            featureVectors[i, 11] = (self.realCandidates[sourceIds[i]] - self.minFeatures[11]) / self.maxFeatures[11] * 10000
            featureVectors[i, 12] = (featureVectors[i, 12] - self.minFeatures[12]) / self.maxFeatures[12] * 10000
            featureVectors[i, 13] = (featureVectors[i, 13] - self.minFeatures[13]) / self.maxFeatures[13] * 10000
            featureVectors[i, 14] = (featureVectors[i, 14] - self.minFeatures[14]) / self.maxFeatures[14] * 10000

        # Compute the mean for each column in each sublist
        sublists = []
        start = 0
        for length in lengths:
            # Extract the sublist
            sublist = featureVectors[start:start + length]

            # Compute the mean for each column (axis=0 means column-wise)
            mean_row = (sublist.mean(axis=0) - self.minClusterFeatures) / self.maxClusterFeatures * 10000

            # Append the mean row
            sublists.append(mean_row)
            start += length

        # Return a 2D NumPy array
        return np.array(sublists)


    def getNoOfBlocks1(self, envelopes):
        blocks = []
        for envelope in envelopes:
            maxX = math.ceil(envelope[2] / self.thetaX)
            maxY = math.ceil(envelope[3] / self.thetaY)
            minX = math.floor(envelope[0] / self.thetaX)
            minY = math.floor(envelope[1] / self.thetaY)
            blocks.append((maxX - minX + 1) * (maxY - minY + 1))
        return blocks


    def verification(self):
        """
        Perform cluster verification using the trained model:
        - Compute probabilities for maxsize number of clusters.
        - Use a sorted list to retain top clusters within the budget.
        - Verify relationships and calculate true positives.
        """

        # Initialize counters and data structures for verification
        totalDecisions, truePositiveDecisions = len(self.verifiedClusters), 0
        sorted_list = SortedList()                              # Sorted list to store Clusters based on their probabilities
        self.minimum_probability_threshold = 0.0                # Initialize probability threshold
        SourceInstanceIndexes, TargetInstanceIndexes = [], []   # Track instance indexes for feature vector calculation
        lengths = []                                            # Lengths hold the number of candidate matches for each target
        Instances = np.array([])

        # Iterate through all target geometries
        counter = 0
        for targetId in range(len(self.targetData)):
          # Retrieve clusters for the current target
          candidateMatches = self.getCandidates(targetId)
          
          # Only consider targets with multiple candidate matches
          if len(candidateMatches) > 1:
            lengths.append(len(candidateMatches))

            # Process each candidate match for the current target
            for candidateMatchId in candidateMatches:
                  totalDecisions += 1

                  # Add candidate and target IDs to their respective lists for feature extraction later
                  SourceInstanceIndexes.append(candidateMatchId)
                  TargetInstanceIndexes.append(targetId)

        print("SourceInstanceIndexes", len(SourceInstanceIndexes))
        print("TargetInstanceIndexes", len(TargetInstanceIndexes))

        # Generate feature vectors for all clusters
        Instances = self.get_feature_vector(SourceInstanceIndexes, TargetInstanceIndexes, lengths)
        counter = 0
        predictions = self.classifier.predict(Instances)  # Predict probabilities using the trained classifier
        self.relations.reset()
        print(len(predictions))

        # Calculate the total number of candidate clusters for debugging purposes
        self.totalCandidateClusters = len(predictions)

        #Calculate the number of clusters will be verified
        maxsize = (self.target_recall+1) * ((1/self.recall_aprox)* (self.detectedQP / self.N)) * self.totalCandidateClusters
        #maxsize = 0.4 * self.totalCandidateClusters

        print("Target Recall =", self.target_recall)
        print("1/self.recall_aprox = ", 1/self.recall_aprox)
        print("DetectedQP =", self.detectedQP)
        print("N = ", self.N)
        print("Total Candidate Clusters", self.totalCandidateClusters)
        print("Maxsize = ", maxsize)

        start = 0
        start_index = 0

        # Process predictions and add them to the sorted list
        for pred, idx in tqdm(zip(predictions,list(self.cluster_data.keys())), desc="Calculating Recall", unit="prediction"):
                weight = float(pred[0])
                if weight >= self.minimum_probability_threshold:
                    sorted_list.add((weight, idx))

        # Sort and process the list
        while sorted_list and len(sorted_list) > self.budget:
            sorted_list.pop(0)  # Maintain budget

        # Reverse the sorted list to process the highest probability clusters first
        for pred, idx_A in reversed(sorted_list):
            similarity_index = self.cluster_data[idx_A]
            if self.relations.verifyRelations(idx_A,similarity_index):
                truePositiveDecisions += 1
            if (math.floor(maxsize) == counter):
                break

            counter += 1

        print("True Positive Decisions\t:\t" + str(truePositiveDecisions))


    def get_best_model(self, x_train, samples=200, h_vals=np.arange(0.001, 0.21, 0.01), seed=42):
        kernels = ['cosine', 'epanechnikov', 'gaussian', 'linear', 'tophat', 'exponential']
        print("Testing {} options with Grid Search".format(len(h_vals)*len(kernels)))
        grid = GridSearchCV(KernelDensity(), {'bandwidth': h_vals, 'kernel': kernels}, cv=LeaveOneOut())
        grid.fit(np.expand_dims(x_train, axis=1))
        print('Best KDE estimator', grid.best_estimator_)
        return grid.best_estimator_

    def find_estimate_threshold(self,model, interpolation_points=1000):
      estimations = []
      for threshold in np.arange(0,1.0,0.00015):
          est = self.compute_estimate_cdf(model, target_range=(0, threshold))
          #print(threshold, 1 - est)
          estimations.append((threshold,1-est))
      self.similarity_index_threshold = self.find_closest(self.similarity_index_range, estimations)
     # print("This is minimum ", self.similarity_index_threshold)

    def compute_estimate_cdf(self,model, target_range=(0, 1), interpolation_points=1000, margin=0.01):
        x_test, log_dens = self.get_logs(model, target_range, interpolation_points, margin)
        probs = np.exp(log_dens)
        auc = np.trapz(probs.ravel(), x_test.ravel())
        return auc

    def generate_test_interval(self,target_range=(0, 1), interpolation_points=1000, margin=0.01):
        start = target_range[0]-target_range[0]*margin
        stop = target_range[1]+target_range[1]*margin
        x_test = np.linspace(start, stop, interpolation_points)[:, np.newaxis]
        return x_test

    def get_logs(self,model, target_range=(0, 1), interpolation_points=1000, margin=0.01):
        x_test = self.generate_test_interval(target_range, interpolation_points, margin)
        log_dens = model.score_samples(x_test)
        return x_test, log_dens

    def find_closest(self,target, tuples_list):
        min_difference = float('inf')
        for first_num, second_num in tuples_list:
            difference = abs(target - second_num)
            if difference < min_difference:
                min_difference = difference
                threshold = first_num
        return threshold
