import csv
import sys
from shapely import wkt
from shapely.geometry import GeometryCollection, Polygon, MultiPolygon
from collections import deque

# Setting the CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

class CsvReader:
    @staticmethod
    def readAllEntities(delimiter, inputFilePath, batch_size=1000):
        loadedEntities = []
        geoCollections = 0
        batch = []

        def process_batch(batch):
            local_entities = []
            local_geo_collections = 0

            for row in batch:
                found_geometry = False
                for column in row:
                    try:
                        geometry = wkt.loads(column)
                        found_geometry = True
                        break
                    except Exception:
                        continue
                if not found_geometry:
                    continue

                if isinstance(geometry, GeometryCollection):
                    local_geo_collections += 1
                else:
                    local_entities.append(geometry)

            return local_entities, local_geo_collections

        # Open and read the CSV file
        with open(inputFilePath, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                batch.append(row)
                if len(batch) >= batch_size:
                    entities, geo_collections = process_batch(batch)
                    loadedEntities.extend(entities)
                    geoCollections += geo_collections
                    batch = []

            # Process the remaining rows in the last batch
            if batch:
                entities, geo_collections = process_batch(batch)
                loadedEntities.extend(entities)
                geoCollections += geo_collections

        print(f"Total entities: {len(loadedEntities)}, Geometry collections: {geoCollections}")
        return loadedEntities

    @staticmethod
    def loadClusterDataToDeque(inputFilePath):
        """
        Loads cluster data from a CSV file into a deque.
        Assumes the CSV file contains two columns: cluster ID (int) and similarity index (float),
        and skips the header row.
        """
        cluster_data = deque()
        with open(inputFilePath, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader, None)  # Skip the header row
            for row in reader:
                cluster_id, similarity_index = int(row[0]), float(row[1])
                cluster_data.append((cluster_id, similarity_index))
        return cluster_data
