import numpy as np
from stl import mesh as stl_mesh
import trimesh
import os
import json

class STLLoader:
    def __init__(self):
        self.models = {}

    def load_stl(self, filepath: str, model_id: str = None) -> dict:
        if model_id is None:
            model_id = os.path.basename(filepath).replace('.stl', '')

        try:
            tm = trimesh.load(filepath, force='mesh')

            stl_data = stl_mesh.Mesh.from_file(filepath)

            # Extract vertices and faces
            vertices = tm.vertices
            faces = tm.faces
            normals = tm.face_normals

            # Bounding box
            bounds = tm.bounds
            bbox_min = bounds[0].tolist()
            bbox_max = bounds[1].tolist()
            dimensions = (bounds[1] - bounds[0]).tolist()

            # Geometric properties
            surface_area = float(tm.area)
            volume = float(abs(tm.volume)) if tm.is_watertight else self._estimate_volume(tm)
            triangle_count = len(faces)
            vertex_count = len(vertices)

            # Curvature statistics (via vertex normals variance)
            vertex_normals = tm.vertex_normals
            curvature_std = float(np.std(vertex_normals, axis=0).mean())
            curvature_mean = float(np.mean(np.abs(vertex_normals), axis=0).mean())

            # Center of mass
            centroid = tm.centroid.tolist()

            model_data = {
                "model_id": model_id,
                "filepath": filepath,
                "trimesh": tm,
                "stl_data": stl_data,
                "vertices": vertices,
                "faces": faces,
                "normals": normals,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "dimensions": dimensions,
                "surface_area": surface_area,
                "volume": volume,
                "triangle_count": triangle_count,
                "vertex_count": vertex_count,
                "curvature_std": curvature_std,
                "curvature_mean": curvature_mean,
                "centroid": centroid,
                "is_watertight": bool(tm.is_watertight),
            }

            self.models[model_id] = model_data
            return model_data

        except Exception as e:
            raise ValueError(f"Failed to load STL '{filepath}': {e}")

    def _estimate_volume(self, tm: trimesh.Trimesh) -> float:
        bounds = tm.bounds
        dims = bounds[1] - bounds[0]
        bbox_vol = float(dims[0] * dims[1] * dims[2])
        # Approximate as ~60% of bounding box for chair-like objects
        return bbox_vol * 0.6

    def load_batch(self, filepaths: list) -> dict:
        """Load multiple STL files."""
        results = {}
        for fp in filepaths:
            model_id = os.path.basename(fp).replace('.stl', '')
            try:
                results[model_id] = self.load_stl(fp, model_id)
                print(f"[STLLoader] Loaded: {model_id}")
            except Exception as e:
                print(f"[STLLoader] Error loading {fp}: {e}")
                results[model_id] = {"error": str(e)}
        return results

    def get_summary(self, model_id: str) -> dict:
        """Return serializable summary of a loaded model."""
        m = self.models.get(model_id)
        if not m:
            return {}
        return {
            "model_id": m["model_id"],
            "bbox_min": m["bbox_min"],
            "bbox_max": m["bbox_max"],
            "dimensions": m["dimensions"],
            "surface_area": round(m["surface_area"], 4),
            "volume": round(m["volume"], 4),
            "triangle_count": m["triangle_count"],
            "vertex_count": m["vertex_count"],
            "curvature_std": round(m["curvature_std"], 6),
            "curvature_mean": round(m["curvature_mean"], 6),
            "centroid": [round(c, 4) for c in m["centroid"]],
            "is_watertight": m["is_watertight"],
        }