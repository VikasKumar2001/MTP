import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import json
import matplotlib.pyplot as plt


class CADSlicingEngine:
    def __init__(self, num_layers: int = 1000):
        self.num_layers = num_layers

    def slice_model(self, model_data: dict) -> dict:
        """
        Slice the STL model along Z axis and extract layer information.
        Returns comprehensive slicing analysis.
        """
        tm = model_data["trimesh"]
        bbox_min = np.array(model_data["bbox_min"])
        bbox_max = np.array(model_data["bbox_max"])

        z_min = bbox_min[2]
        z_max = bbox_max[2]
        z_range = z_max - z_min

        z_levels = np.linspace(z_min + 0.01 * z_range, z_max - 0.01 * z_range, self.num_layers)
        layer_height = z_range / self.num_layers

        layers = []
        areas = []
        perimeters = []
        centroids_x = []
        centroids_y = []
        occupancy = []

        for i, z in enumerate(z_levels):
            try:
                section = tm.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
                if section is None:
                    layers.append(None)
                    areas.append(0)
                    perimeters.append(0)
                    centroids_x.append(0)
                    centroids_y.append(0)
                    occupancy.append(0)
                    continue

                # Convert path to 2D
                path2d, _ = section.to_planar()
                polygons = []
                for entity in path2d.entities:
                    if len(entity.points) >= 3:
                        pts = path2d.vertices[entity.points]
                        try:
                            poly = Polygon(pts)
                            if poly.is_valid and poly.area > 0:
                                polygons.append(poly)
                        except Exception:
                            pass

                if polygons:
                    combined = unary_union(polygons)
                    area = combined.area
                    perimeter = combined.length
                    cx, cy = combined.centroid.x, combined.centroid.y
                else:
                    area = 0
                    perimeter = 0
                    cx, cy = 0, 0

                areas.append(float(area))
                perimeters.append(float(perimeter))
                centroids_x.append(float(cx))
                centroids_y.append(float(cy))

                # Occupancy = ratio of area to bounding box footprint
                footprint = (bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1])
                occ = area / footprint if footprint > 0 else 0
                occupancy.append(float(min(occ, 1.0)))

                layers.append({
                    "layer_index": i,
                    "z": float(z),
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                    "occupancy": float(min(occ, 1.0)),
                })

            except Exception:
                layers.append(None)
                areas.append(0)
                perimeters.append(0)
                centroids_x.append(0)
                centroids_y.append(0)
                occupancy.append(0)

        # ---- Structural Feature Extraction from Slices ----
        areas_np = np.array(areas)
        occ_np = np.array(occupancy)

        # Seat height: largest cross-sectional area peak in lower 60%
        lower_idx = int(self.num_layers * 0.6)
        lower_areas = areas_np[:lower_idx]
        seat_layer_idx = int(np.argmax(lower_areas)) if lower_areas.max() > 0 else lower_idx // 2
        seat_height_mm = float(z_levels[seat_layer_idx] - z_min)

        # Backrest detection: significant area in upper 40%
        upper_areas = areas_np[lower_idx:]
        backrest_present = bool(upper_areas.max() > areas_np.mean() * 0.3)
        backrest_layer_idx = lower_idx + int(np.argmax(upper_areas)) if backrest_present else -1

        # Backrest angle estimation (centroid drift)
        if backrest_present and backrest_layer_idx > 0:
            cx_lower = centroids_x[seat_layer_idx] if seat_layer_idx < len(centroids_x) else 0
            cx_upper = centroids_x[backrest_layer_idx] if backrest_layer_idx < len(centroids_x) else 0
            cy_lower = centroids_y[seat_layer_idx] if seat_layer_idx < len(centroids_y) else 0
            cy_upper = centroids_y[backrest_layer_idx] if backrest_layer_idx < len(centroids_y) else 0
            drift = np.sqrt((cx_upper - cx_lower)**2 + (cy_upper - cy_lower)**2)
            height_diff = float(z_levels[backrest_layer_idx] - z_levels[seat_layer_idx])
            backrest_angle = float(np.degrees(np.arctan2(drift, height_diff))) if height_diff > 0 else 0
        else:
            backrest_angle = 0.0

        # Armrest detection: local area bulge at seat mid-height
        mid_low = int(self.num_layers * 0.35)
        mid_high = int(self.num_layers * 0.55)
        mid_areas = areas_np[mid_low:mid_high]
        armrest_present = bool(mid_areas.max() > lower_areas.max() * 0.7) if lower_areas.max() > 0 else False

        # Leg configuration: base area pattern in lowest 15%
        base_idx = int(self.num_layers * 0.15)
        base_areas = areas_np[:base_idx]
        base_occ = occ_np[:base_idx]

        # Multiple legs: high variance in base occupancy
        leg_variance = float(np.var(base_occ)) if len(base_occ) > 0 else 0
        if leg_variance > 0.02:
            leg_config = "4-leg"
        elif base_areas.mean() < areas_np.mean() * 0.3:
            leg_config = "pedestal"
        else:
            leg_config = "sled"

        # Stability metric: ratio of base area to seat area
        base_mean_area = float(base_areas.mean()) if len(base_areas) > 0 else 0
        seat_area = float(areas_np[seat_layer_idx]) if areas_np[seat_layer_idx] > 0 else 1
        stability = float(min(base_mean_area / seat_area, 1.0)) if seat_area > 0 else 0

        # Ergonomic score
        ergo = 0.0
        if backrest_present:
            ergo += 0.3
        if armrest_present:
            ergo += 0.2
        if 380 <= seat_height_mm <= 520:
            ergo += 0.3
        ergo += stability * 0.2
        ergonomic_score = round(float(min(ergo, 1.0)), 3)

        # Density map (simplified as occupancy per layer)
        density_map = [round(o, 4) for o in occupancy]

        return {
            "num_layers": self.num_layers,
            "z_min": float(z_min),
            "z_max": float(z_max),
            "z_range": float(z_range),
            "layer_height": float(layer_height),
            "layers": layers,
            "areas": [round(a, 4) for a in areas],
            "perimeters": [round(p, 4) for p in perimeters],
            "centroids_x": [round(c, 4) for c in centroids_x],
            "centroids_y": [round(c, 4) for c in centroids_y],
            "occupancy": density_map,
            "density_map": density_map,
            # Structural features
            "seat_height_mm": round(seat_height_mm, 1),
            "seat_area": round(seat_area, 4),
            "backrest_present": backrest_present,
            "backrest_angle_deg": round(backrest_angle, 2),
            "armrest_present": armrest_present,
            "leg_config": leg_config,
            "leg_variance": round(leg_variance, 6),
            "stability_score": round(stability, 4),
            "ergonomic_score": ergonomic_score,
            "mean_area": round(float(areas_np.mean()), 4),
            "max_area": round(float(areas_np.max()), 4),
            "area_std": round(float(areas_np.std()), 4),
        }

    def get_slice_signature(self, slicing_result: dict) -> list:
        """Create a compact numerical signature of the slice profile."""
        areas = slicing_result["areas"]
        max_a = max(areas) if max(areas) > 0 else 1
        # Normalize and sample 20 values
        normalized = [round(a / max_a, 4) for a in areas]
        step = max(1, len(normalized) // 20)
        return normalized[::step][:20]
    
    def plot_slice_graph(self, slicing_result,save_path):
    
        z_values = [layer["z"] for layer in slicing_result["layers"] if layer]
        areas = [layer["area"] for layer in slicing_result["layers"] if layer]
        perimeters = [layer["perimeter"] for layer in slicing_result["layers"] if layer]

        plt.figure(figsize=(12, 6))

        plt.plot(z_values, areas, label="Area")
        plt.plot(z_values, perimeters, label="Perimeter")

        plt.xlabel("Z Height")
        plt.ylabel("Value")
        plt.title("Slice Analysis Graph")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()