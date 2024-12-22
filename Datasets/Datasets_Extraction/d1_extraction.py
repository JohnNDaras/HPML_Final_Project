import csv
import sys
import multiprocessing
from shapely import wkt
from shapely.geometry import GeometryCollection

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
    # Paths to input and output files
    main_dir = '../content/drive/MyDrive/hpml_final_project/D1/'
    sourceFilePath = "../content/drive/MyDrive/s1/sourceSample.tsv"
    targetFilePath = "../content/drive/MyDrive/s1/targetSample.tsv"
    sourceOutputFile = main_dir + "SourceDataset.csv"
    targetOutputFile =  main_dir + "TargetDataset.csv"


        # Multiprocessing for both datasets
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.starmap(CsvReader.readAllEntities, [
            ("\t", sourceFilePath),
            ("\t", targetFilePath)
        ])

    sourceData, targetData = results
  
    # Write the two lists to separate CSV files
    write_polygon_strings_to_csv(sourceData, sourceOutputFile)
    write_polygon_strings_to_csv(targetData, targetOutputFile)
    print("Processing complete for both datasets.")