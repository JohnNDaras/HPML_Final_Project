import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate
from shapely.set_operations import intersection_all, union_all
from shapely.strtree import STRtree

class ShapeSimilarity:

    def __init__(self):
        pass

    # Function to center polygons at the origin
    def center_polygons(self, polygons):
        centered_polygons = []
        for polygon in polygons:
            centroid = polygon.centroid
            centered_polygon = translate(polygon, xoff=-centroid.x, yoff=-centroid.y)
            centered_polygons.append(centered_polygon)
        return np.array(centered_polygons)

    # Precompute polygon properties and store them for efficiency
    def _precompute_properties(self, polygons):
        self.polygons = self.center_polygons(polygons)  # Center all polygons before computing properties
        self.num_polygons = len(self.polygons)

        # Precompute properties
        self.areas = np.array([polygon.area for polygon in self.polygons])
        self.perimeters = np.array([polygon.length for polygon in self.polygons])
        self.bboxes = np.array([polygon.bounds for polygon in self.polygons])
        self.fourier_descriptors = np.array([self._fourier_descriptor(polygon) for polygon in self.polygons])

    # Fourier Descriptor for a polygon
    def _fourier_descriptor(self, polygon, num_points=128):
        coords = np.array(polygon.exterior.coords)
        t = np.linspace(0, 1, len(coords))
        resampled_t = np.linspace(0, 1, num_points)
        resampled_coords = np.column_stack((
            np.interp(resampled_t, t, coords[:, 0]),
            np.interp(resampled_t, t, coords[:, 1])
        ))
        complex_coords = resampled_coords[:, 0] + 1j * resampled_coords[:, 1]
        fourier_transform = np.fft.fft(complex_coords)
        return np.abs(fourier_transform / np.abs(fourier_transform[1]))  # Normalize

    # Jaccard Similarity
    def jaccard_similarity(self, A, B):
        intersection_area = A.intersection(B).area
        union_area = A.union(B).area
        return intersection_area / union_area if union_area != 0 else 0

    # Area Similarity
    def area_similarity(self, idx_A, idx_B):
        intersection_area = self.polygons[idx_A].intersection(self.polygons[idx_B]).area
        return (2 * intersection_area) / (self.areas[idx_A] + self.areas[idx_B]) if self.areas[idx_A] + self.areas[idx_B] > 0 else 0

    # Curvature Similarity
    def curvature_similarity(self, idx_A, idx_B):
        num_vertices_A = len(self.polygons[idx_A].exterior.coords)
        num_vertices_B = len(self.polygons[idx_B].exterior.coords)
        return np.exp(-abs(num_vertices_A - num_vertices_B) / max(num_vertices_A, num_vertices_B))

    # Fourier Descriptor Similarity
    def fourier_descriptor_similarity(self, idx_A, idx_B):
        return 1 / (1 + np.linalg.norm(self.fourier_descriptors[idx_A] - self.fourier_descriptors[idx_B]))

    # Aspect Ratio Similarity
    def aspect_ratio_similarity(self, bbox_A, bbox_B):
        aspect_ratio_A = (bbox_A[2] - bbox_A[0]) / (bbox_A[3] - bbox_A[1]) if bbox_A[3] != bbox_A[1] else 0
        aspect_ratio_B = (bbox_B[2] - bbox_B[0]) / (bbox_B[3] - bbox_B[1]) if bbox_B[3] != bbox_B[1] else 0
        return 1 / (1 + abs(aspect_ratio_A - aspect_ratio_B))

    # Perimeter Similarity
    def perimeter_similarity(self, idx_A, idx_B):
        return 1 / (1 + abs(self.perimeters[idx_A] - self.perimeters[idx_B]))

    # Bounding Box Distance
    def bounding_box_distance(self, idx_A, idx_B):
        center_A = ((self.bboxes[idx_A][0] + self.bboxes[idx_A][2]) / 2, (self.bboxes[idx_A][1] + self.bboxes[idx_A][3]) / 2)
        center_B = ((self.bboxes[idx_B][0] + self.bboxes[idx_B][2]) / 2, (self.bboxes[idx_B][1] + self.bboxes[idx_B][3]) / 2)
        dist_centers = np.linalg.norm(np.array(center_A) - np.array(center_B))
        return 1 / (1 + dist_centers)

    # Polygon Circularity Similarity
    def polygon_circularity_similarity(self, idx_A, idx_B):
        circularity_A = (4 * np.pi * self.areas[idx_A]) / (self.perimeters[idx_A] ** 2) if self.perimeters[idx_A] != 0 else 0
        circularity_B = (4 * np.pi * self.areas[idx_B]) / (self.perimeters[idx_B] ** 2) if self.perimeters[idx_B] != 0 else 0
        return 1 / (1 + abs(circularity_A - circularity_B))

    # Combined similarity calculation for each unique pair
    def combined_similarity(self, idx_A, idx_B, w_jaccard=0.125, w_area=0.125, w_curvature=0.125, w_fourier=0.125,
                            w_aspect_ratio=0.125, w_perimeter=0.125, w_bbox=0.125, w_circularity=0.125):

        jaccard_sim = self.jaccard_similarity(self.polygons[idx_A], self.polygons[idx_B])
        area_sim = self.area_similarity(idx_A, idx_B)
        curvature_sim = self.curvature_similarity(idx_A, idx_B)
        fourier_sim = self.fourier_descriptor_similarity(idx_A, idx_B)
        aspect_ratio_sim = self.aspect_ratio_similarity(self.bboxes[idx_A], self.bboxes[idx_B])
        perimeter_sim = self.perimeter_similarity(idx_A, idx_B)
        bbox_dist = self.bounding_box_distance(idx_A, idx_B)
        circularity_sim = self.polygon_circularity_similarity(idx_A, idx_B)

        return (w_jaccard * jaccard_sim +
                w_area * area_sim +
                w_curvature * curvature_sim +
                w_fourier * fourier_sim +
                w_aspect_ratio * aspect_ratio_sim +
                w_perimeter * perimeter_sim +
                w_bbox * bbox_dist +
                w_circularity * circularity_sim) * 100

    # Calculate average similarity for all unique pairs
    def calculate_similarity_all_pairs(self, polygons_array):
        self._precompute_properties(polygons_array)
        similarity_scores = []

        for i in range(self.num_polygons):
            for j in range(i + 1, self.num_polygons):
                similarity_score = self.combined_similarity(i, j)
                similarity_scores.append(similarity_score)

        return np.mean(similarity_scores) if similarity_scores else 0
