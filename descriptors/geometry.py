import numpy as np

class GeometryDescriptor:
    """
    Milestone 4: Extracts core geometric features into a normalized continuous vector.
    Computes volume, surface area, bounding box extents, aspect ratios, and PCA eigenvalues.
    """
    def extract(self, model_data: dict) -> dict:
        # 1. Basic properties
        dims = model_data.get("dimensions", [1.0, 1.0, 1.0])
        vol = model_data.get("volume", 0.0)
        area = model_data.get("surface_area", 0.0)
        curvature = model_data.get("curvature_std", 0.0)
        
        # 2. Aspect Ratios (Prevent division by zero)
        dx = max(dims[0], 1e-6)
        dy = max(dims[1], 1e-6)
        dz = max(dims[2], 1e-6)
        
        aspect_xy = dx / dy
        aspect_xz = dx / dz
        aspect_yz = dy / dz

        # 3. PCA (Principal Component Analysis)
        # Calculates the spread of the geometry along its principal axes
        pca_features = [0.0, 0.0, 0.0]
        if "vertices" in model_data and len(model_data["vertices"]) > 0:
            vertices = np.array(model_data["vertices"])
            
            # Center the vertices around the origin
            centroid = np.mean(vertices, axis=0)
            centered = vertices - centroid
            
            # Compute covariance matrix and eigenvalues
            cov_matrix = np.cov(centered, rowvar=False)
            eigenvalues, _ = np.linalg.eigh(cov_matrix)
            
            # Sort eigenvalues in descending order
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Normalize eigenvalues to describe the relative shape distribution
            max_eig = max(eigenvalues[0], 1e-6)
            pca_features = [
                float(eigenvalues[0] / max_eig),
                float(eigenvalues[1] / max_eig),
                float(eigenvalues[2] / max_eig)
            ]

        # 4. Build the normalized geometry vector (16-Dimensional)
        vector = [
            min(vol / 1e7, 1.0),         # Normalized Volume
            min(area / 1e6, 1.0),        # Normalized Area
            min(curvature * 10, 1.0),    # Scaled Curvature
            dims[0] / 1000.0,            # Normalized X Dim
            dims[1] / 1000.0,            # Normalized Y Dim
            dims[2] / 1000.0,            # Normalized Z Dim
            min(aspect_xy, 10.0) / 10.0, # Scaled Aspect XY
            min(aspect_xz, 10.0) / 10.0, # Scaled Aspect XZ
            min(aspect_yz, 10.0) / 10.0, # Scaled Aspect YZ
            pca_features[0],             # PCA 1 (always 1.0)
            pca_features[1],             # PCA 2 (relative variance)
            pca_features[2],             # PCA 3 (relative variance)
        ]
        
        # Pad to exactly 16 dimensions for consistency in the fusion layer
        while len(vector) < 16:
            vector.append(0.0)
            
        return {
            "geometry_vector": [round(v, 5) for v in vector]
        }