# README

## Overview
This directory is designed to extract and process geospatial datasets for use in a larger experimental framework. These datasets are sourced from well-known repositories such as the US Census Bureau's TIGER/Line Shapefiles and OpenStreetMap (OSM). The main objective is to transform raw geospatial data into polygonal geometries for further analysis.

## Datasets
The datasets utilized in this project are:

### US Census Bureau TIGER/Line Shapefiles
1. **AREAWATER**: Polygonal geometries representing water areas such as lakes, reservoirs, and ponds.
2. **LINEARWATER**: Linear geometries representing hydrographic features like rivers, streams, and canals.

### OpenStreetMap (OSM) Datasets
1. **Lakes**: Polygonal geometries of lakes extracted from OSM.
2. **Parks**: Polygonal geometries of park areas extracted from OSM.
3. **Roads**: Linear geometries of roads extracted from OSM.

## Features
- **Batch Processing**: Efficient handling of large datasets by processing data in batches.
- **Parallel Processing**: Leverages multiprocessing to expedite data extraction and transformation.
- **Configurable Extraction**: Allows setting limits on the number of entities to extract, providing scalability and flexibility.

## Output
The output consists of CSV files containing the string representation of processed polygonal geometries. Each dataset is exported to its respective CSV file and is ready for further analysis.

## Usage
### Key Parameters
- **Input File Paths**: Specifies the location of the raw datasets.
- **Output File Paths**: Defines the destination for processed datasets.
- **Delimiter**: Indicates the delimiter used in the input files.
- **Entity Limits**: Configurable limit for the number of geometries to process.

### Execution
1. Configure the file paths and parameters in the main script.
2. Run the script to process the datasets.
3. The results will be saved as CSV files in the specified output directories.

## Dataset Summary Table

| Dataset | Source Geometries | Target Geometries | Total Clusters | Average Cluster Size |
|---------|-------------------|-------------------|----------------|----------------------|
| D1      | 229,276           | 583,833           | 295,481        | ~6 polygons          |
| D2      | 210,483           | 2,898,899         | 654,196        | ~13 polygons         |
| D3      | 200,294           | 7,392,699         | 1,324,980      | ~34 polygons         |





