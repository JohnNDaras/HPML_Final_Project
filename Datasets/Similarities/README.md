# Similarity Results Processor

## Overview
This project provides a utility to calculate and analyze spatial similarities between geometries in three datasets. It processes the datasets to determine similarities and saves the results in respective output files. These files are intended for use in the main program located in the `Dynamic_Similarity_Algorithm` directory.

## Purpose
The primary purpose of this code is to:
1. Process three datasets containing spatial geometries.
2. Calculate similarity metrics between geometries in the source and target datasets.
3. Save the computed similarities to respective output files.
4. Provide these similarity results as input for further processing in the `Dynamic_Similarity_Algorithm` project.

## Output
- The output files contain calculated similarity metrics and are stored in the specified output directory. These files are structured to integrate seamlessly with the main codebase in the `Dynamic_Similarity_Algorithm` directory.

## Similarity Range Details

### D1
| Similarity Range | Number of Clusters |
|------------------|---------------------|
| 0-10%           | 0                   |
| 10-20%          | 0                   |
| 20-30%          | 68                  |
| 30-40%          | 10,864              |
| 40-50%          | 64,553              |
| 50-60%          | 121,251             |
| 60-70%          | 68,206              |
| 70-80%          | 27,246              |
| 80-90%          | 3,215               |
| 90-100%         | 65                  |

### D2
| Similarity Range | Number of Clusters |
|------------------|---------------------|
| 0-10%           | 0                   |
| 10-20%          | 0                   |
| 20-30%          | 0                   |
| 30-40%          | 3,299               |
| 40-50%          | 49,966              |
| 50-60%          | 289,280             |
| 60-70%          | 254,385             |
| 70-80%          | 45,820              |
| 80-90%          | 9,243               |
| 90-100%         | 2,181               |

### D3
| Similarity Range | Number of Clusters |
|------------------|---------------------|
| 0-10%           | 0                   |
| 10-20%          | 0                   |
| 20-30%          | 0                   |
| 30-40%          | 4,143               |
| 40-50%          | 93,512              |
| 50-60%          | 501,648             |
| 60-70%          | 605,463             |
| 70-80%          | 103,164             |
| 80-90%          | 13,195              |
| 90-100%         | 3,582               |

