from shapely import relate
from shapely import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely import get_num_coordinates

class RelatedGeometries:
    def __init__(self, qualifyingClusters, average_similarity):
        self.pgr = 0
        self.exceptions = 0
        self.detectedLinks = 0
        self.verifiedClusters = 0
        self.qualifyingClusters = qualifyingClusters
        self.average_similarity = average_similarity
        self.interlinkedGeometries = 0
        self.continuous_unrelated_Clusters = 0
        self.violations = 0

        # Lists to store counts of geometries by similarity ranges (0-100%)
        self.similarity_0_10 = []
        self.similarity_10_20 = []
        self.similarity_20_30 = []
        self.similarity_30_40 = []
        self.similarity_40_50 = []
        self.similarity_50_60 = []
        self.similarity_60_70 = []
        self.similarity_70_80 = []
        self.similarity_80_90 = []
        self.similarity_90_100 = []

    # Methods to add geometries to the corresponding similarity range
    def addSimilarity(self, cluster, similarity):
        if 0 <= similarity < 10:
            self.similarity_0_10.append(cluster)
        elif 10 <= similarity < 20:
            self.similarity_10_20.append(cluster)
        elif 20 <= similarity < 30:
            self.similarity_20_30.append(cluster)
        elif 30 <= similarity < 40:
            self.similarity_30_40.append(cluster)
        elif 40 <= similarity < 55:
            self.similarity_40_50.append(cluster)
        elif 55 <= similarity < 60:
            self.similarity_50_60.append(cluster)
        elif 60 <= similarity < 70:
            self.similarity_60_70.append(cluster)
        elif 70 <= similarity < 80:
            self.similarity_70_80.append(cluster)
        elif 80 <= similarity < 90:
            self.similarity_80_90.append(cluster)
        elif 90 <= similarity <= 100:
            self.similarity_90_100.append(cluster)

    # Get counts of Clusters in each similarity range
    def getNoOfClustersInRange(self, lower_bound, upper_bound):
        if lower_bound == 0 and upper_bound == 10:
            return len(self.similarity_0_10)
        elif lower_bound == 10 and upper_bound == 20:
            return len(self.similarity_10_20)
        elif lower_bound == 20 and upper_bound == 30:
            return len(self.similarity_20_30)
        elif lower_bound == 30 and upper_bound == 40:
            return len(self.similarity_30_40)
        elif lower_bound == 40 and upper_bound == 50:
            return len(self.similarity_40_50)
        elif lower_bound == 50 and upper_bound == 60:
            return len(self.similarity_50_60)
        elif lower_bound == 60 and upper_bound == 70:
            return len(self.similarity_60_70)
        elif lower_bound == 70 and upper_bound == 80:
            return len(self.similarity_70_80)
        elif lower_bound == 80 and upper_bound == 90:
            return len(self.similarity_80_90)
        elif lower_bound == 90 and upper_bound == 100:
            return len(self.similarity_90_100)
        else:
            return 0

    def reset(self):
        self.pgr = 0
        self.exceptions = 0
        self.detectedLinks = 0
        self.verifiedClusters = 0
        self.interlinkedGeometries = 0

        # Clear similarity ranges
        self.similarity_0_10.clear()
        self.similarity_10_20.clear()
        self.similarity_20_30.clear()
        self.similarity_30_40.clear()
        self.similarity_40_50.clear()
        self.similarity_50_60.clear()
        self.similarity_60_70.clear()
        self.similarity_70_80.clear()
        self.similarity_80_90.clear()
        self.similarity_90_100.clear()

    def print(self):
        print("Qualifying Clusters:\t", str(self.qualifyingClusters))
        print("Exceptions:\t", str(self.exceptions))
        print("Detected Links:\t", str(self.detectedLinks))
        print("Interlinked geometries:\t", str(self.interlinkedGeometries))
        print("Clusters in 0-10% similarity range:\t", str(len(self.similarity_0_10)))
        print("Clusters in 10-20% similarity range:\t", str(len(self.similarity_10_20)))
        print("Clusters in 20-30% similarity range:\t", str(len(self.similarity_20_30)))
        print("Clusters in 30-40% similarity range:\t", str(len(self.similarity_30_40)))
        print("Clusters in 40-50% similarity range:\t", str(len(self.similarity_40_50)))
        print("Clusters in 50-60% similarity range:\t", str(len(self.similarity_50_60)))
        print("Clusters in 60-70% similarity range:\t", str(len(self.similarity_60_70)))
        print("Clusters in 70-80% similarity range:\t", str(len(self.similarity_70_80)))
        print("Clusters in 80-90% similarity range:\t", str(len(self.similarity_80_90)))
        print("Clusters in 90-100% similarity range:\t", str(len(self.similarity_90_100)))
        if self.qualifyingClusters != 0:
            print("Recall", str((self.interlinkedGeometries / float(self.qualifyingClusters))))
        if self.verifiedClusters != 0:
            print("Precision", str((self.interlinkedGeometries / self.verifiedClusters)))
        if self.qualifyingClusters != 0 and self.verifiedClusters != 0:
            print("Progressive Geometry Recall", str(self.pgr / self.qualifyingClusters / self.verifiedClusters))
        print("Verified Clusters", str(self.verifiedClusters))

    def verifyRelations(self, cluster, similarity):
        related = False
        self.verifiedClusters += 1

        if similarity > self.average_similarity :
            related = True
            self.interlinkedGeometries += 1
            self.pgr += self.interlinkedGeometries
            self.continuous_unrelated_Clusters = 0
        else:
            self.continuous_unrelated_Clusters += 1

        # Add the pair to the corresponding similarity range
        #self.addSimilarity(cluster, similarity)
        if 0 <= similarity < 10:
            self.similarity_0_10.append(cluster)
        elif 10 <= similarity < 20:
            self.similarity_10_20.append(cluster)
        elif 20 <= similarity < 30:
            self.similarity_20_30.append(cluster)
        elif 30 <= similarity < 40:
            self.similarity_30_40.append(cluster)
        elif 40 <= similarity < 50:
            self.similarity_40_50.append(cluster)
        elif 50 <= similarity < 60:
            self.similarity_50_60.append(cluster)
        elif 60 <= similarity < 70:
            self.similarity_60_70.append(cluster)
        elif 70 <= similarity < 80:
            self.similarity_70_80.append(cluster)
        elif 80 <= similarity < 90:
            self.similarity_80_90.append(cluster)
        elif 90 <= similarity <= 100:
            self.similarity_90_100.append(cluster)

        return related
