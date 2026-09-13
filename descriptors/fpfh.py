import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

class FPFHDescriptor:
    """
    Milestone 6: Extracts FPFH (Fast Point Feature Histograms).
    Pools point-level features into a global 33-D vector.
    """
    def __init__(self, target_dim: int = 33):
        self.target_dim = target_dim

    def extract(self, model_data: dict) -> dict:
        if not HAS_O3D:
            print("[WARNING] Open3D not found. Returning zero vector. Install with: pip install open3d")
            return {"fpfh_vector": [0.0] * self.target_dim}

        vertices = np.array(model_data.get("vertices", []))
        if len(vertices) == 0:
            return {"fpfh_vector": [0.0] * self.target_dim}

        try:
            # 1. Initialize Open3D Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            # 2. Dynamic Scaling
            # We scale the search radii based on the bounding box to make it scale-invariant
            dims = model_data.get("dimensions", [1.0, 1.0, 1.0])
            max_dim = max(max(dims), 1e-6)
            voxel_size = max_dim / 50.0  

            # Downsample for faster computation and uniform density
            pcd = pcd.voxel_down_sample(voxel_size)

            # 3. Estimate Normals
            radius_normal = voxel_size * 2
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )

            # 4. Compute FPFH
            radius_feature = voxel_size * 5
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
            )

            # fpfh.data has shape (33, num_points). 
            # 5. Global Pooling (Average)
            fpfh_data = np.asarray(fpfh.data)
            global_fpfh = np.mean(fpfh_data, axis=1)

            # Normalize the pooled histogram so it sums to 1
            total = np.sum(global_fpfh)
            if total > 0:
                global_fpfh = global_fpfh / total

            result = list(global_fpfh)

        except Exception as e:
            print(f"[ERROR] FPFH computation failed: {e}")
            result = [0.0] * self.target_dim

        return {
            "fpfh_vector": [round(float(v), 6) for v in result]
        }