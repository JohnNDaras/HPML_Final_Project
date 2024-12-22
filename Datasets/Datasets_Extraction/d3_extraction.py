import csv
import sys
from shapely import wkt
from shapely.geometry import GeometryCollection, Polygon, MultiPolygon
import multiprocessing


# Setting the CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


class CsvReader1:
    @staticmethod
    def readAllEntities(delimiter, inputFilePath, max_entities, batch_size=1000):
        loadedEntities = []
        geoCollections = 0
        batch = []

        def process_batch(batch):
            local_entities = []
            local_geo_collections = 0

            for row in batch:
                found_geometry = False
                geometry = None
                for column in row:
                    try:
                        geometry = wkt.loads(column)
                        found_geometry = True
                        break
                    except Exception:
                        continue
                if not found_geometry:
                    continue

                # Check if the geometry is a GeometryCollection
                if isinstance(geometry, GeometryCollection):
                    local_geo_collections += 1
                # Add only Polygon or MultiPolygon to the list
                elif isinstance(geometry, Polygon) and geometry.is_valid and not geometry.is_empty:
                    local_entities.append(geometry)

                    # Stop adding more entities if the limit is reached
                    if max_entities is not None and len(loadedEntities) + len(local_entities) >= max_entities:
                        break

            return local_entities, local_geo_collections

        # Open and read the CSV file
        with open(inputFilePath, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                batch.append(row)
                if len(batch) >= batch_size:
                    entities, geo_collections = process_batch(batch)
                    loadedEntities.extend(entities[:max_entities - len(loadedEntities)] if max_entities else entities)
                    geoCollections += geo_collections
                    batch = []

                    # Stop processing if the limit is reached
                    if max_entities is not None and len(loadedEntities) >= max_entities:
                        return loadedEntities

            # Process the remaining rows in the last batch
            if batch and (max_entities is None or len(loadedEntities) < max_entities):
                entities, geo_collections = process_batch(batch)
                loadedEntities.extend(entities[:max_entities - len(loadedEntities)] if max_entities else entities)
                geoCollections += geo_collections

        print(f"Total entities (polygons): {len(loadedEntities)}, Geometry collections: {geoCollections}")
        return loadedEntities


class CsvReader2:
    @staticmethod
    def readAllEntities(delimiter, inputFilePath, max_entities, batch_size=1000):
        loadedEntities = []
        geoCollections = 0
        batch = []

        def process_batch(batch):
            local_entities = []
            local_geo_collections = 0

            for row in batch:
                found_geometry = False
                geometry = None
                for column in row:
                    try:
                        geometry = wkt.loads(column)
                        found_geometry = True
                        break
                    except Exception:
                        continue
                if not found_geometry:
                    continue

                # Check if the geometry is a GeometryCollection
                if isinstance(geometry, GeometryCollection):
                    local_geo_collections += 1
                # Add only Polygon or MultiPolygon to the list
                elif geometry.is_valid and not geometry.is_empty:
                    local_entities.append(geometry)

                    # Stop adding more entities if the limit is reached
                    if max_entities is not None and len(loadedEntities) + len(local_entities) >= max_entities:
                        break

            return local_entities, local_geo_collections

        # Open and read the CSV file
        with open(inputFilePath, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                batch.append(row)
                if len(batch) >= batch_size:
                    entities, geo_collections = process_batch(batch)
                    loadedEntities.extend(entities[:max_entities - len(loadedEntities)] if max_entities else entities)
                    geoCollections += geo_collections
                    batch = []

                    # Stop processing if the limit is reached
                    if max_entities is not None and len(loadedEntities) >= max_entities:
                        return loadedEntities

            # Process the remaining rows in the last batch
            if batch and (max_entities is None or len(loadedEntities) < max_entities):
                entities, geo_collections = process_batch(batch)
                loadedEntities.extend(entities[:max_entities - len(loadedEntities)] if max_entities else entities)
                geoCollections += geo_collections

        print(f"Total entities (polygons): {len(loadedEntities)}, Geometry collections: {geoCollections}")
        return loadedEntities


def run_csv_reader(reader_class, delimiter, file_path, max_entities):
    """Worker function to run a specific CsvReader class."""
    return reader_class.readAllEntities(delimiter, file_path, max_entities)


def write_polygon_strings_to_csv(polygons, filename):
    """
    Writes the string representation of Shapely polygons to a CSV file.

    :param polygons: List of Shapely Polygon objects.
    :param filename: Output CSV filename.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for polygon in polygons:
            writer.writerow([str(polygon)])


if __name__ == "__main__":
    sourceFilePath = "../content/drive/MyDrive/parks"
    targetFilePath = "../content/drive/MyDrive/roads"
    main_dir = "../content/drive/MyDrive/hpml_final_project/D3/"
    sourceOutputFile = main_dir + "SourceDataset.csv"
    targetOutputFile = main_dir + "TargetDataset.csv"
    sourceMaxEntities = 200294
    targetMaxEntities = 7392699

    # Create a pool of workers
    with multiprocessing.Pool(processes=2) as pool:
        # Define tasks
        source_task = pool.apply_async(run_csv_reader, (CsvReader1, "\t", sourceFilePath, sourceMaxEntities))
        target_task = pool.apply_async(run_csv_reader, (CsvReader2, "\t", targetFilePath, targetMaxEntities))

        # Wait for tasks to complete and retrieve results
        sourceData = source_task.get()
        targetData = target_task.get()

    print("Source geometries:", len(sourceData))
    print("Target geometries:", len(targetData))

    write_polygon_strings_to_csv(sourceData, sourceOutputFile)
    write_polygon_strings_to_csv(targetData, targetOutputFile)
