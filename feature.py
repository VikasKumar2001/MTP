import numpy as np
import json


class FeatureIntelligenceSystem:
    def __init__(self):
        self.chair_type_rules = self._build_rules()

    def _build_rules(self):
        """Rule-based classification for chair types."""
        return {
            "wheelchair": {
                "seat_height_mm": (450, 520),
                "armrest_present": True,
                "leg_config": ["sled", "pedestal"],
                "stability_score": (0.6, 1.0),
                "keywords": ["wheel", "mobility"],
            },
            "office": {
                "seat_height_mm": (420, 530),
                "armrest_present": True,
                "leg_config": ["pedestal", "5-wheel"],
                "ergonomic_score": (0.6, 1.0),
            },
            "hospital": {
                "seat_height_mm": (450, 510),
                "armrest_present": True,
                "stability_score": (0.5, 1.0),
            },
            "dining": {
                "seat_height_mm": (440, 500),
                "armrest_present": False,
                "leg_config": ["4-leg"],
                "ergonomic_score": (0.3, 0.65),
            },
            "lounge": {
                "seat_height_mm": (350, 430),
                "backrest_present": True,
                "backrest_angle_deg": (15, 45),
            },
            "stool": {
                "backrest_present": False,
                "seat_height_mm": (550, 800),
            },
        }

    def classify_chair(self, slicing_data: dict, geometry_data: dict) -> str:
        """Classify chair type from features."""
        seat_h = slicing_data.get("seat_height_mm", 0)
        armrest = slicing_data.get("armrest_present", False)
        backrest = slicing_data.get("backrest_present", True)
        leg_cfg = slicing_data.get("leg_config", "4-leg")
        ergo = slicing_data.get("ergonomic_score", 0.5)
        stability = slicing_data.get("stability_score", 0.5)
        backrest_angle = slicing_data.get("backrest_angle_deg", 10)
        dims = geometry_data.get("dimensions", [500, 500, 900])
        height = dims[2] if len(dims) > 2 else 900

        scores = {}

        # Stool
        if not backrest and height > 550:
            scores["stool"] = 0.8
        elif not backrest:
            scores["stool"] = 0.5

        # Lounge
        if backrest_angle > 15 and seat_h < 440:
            scores["lounge"] = 0.75

        # Dining
        if not armrest and 440 <= seat_h <= 500 and leg_cfg == "4-leg":
            scores["dining"] = 0.8

        # Office
        if armrest and ergo >= 0.6 and (leg_cfg in ["pedestal", "5-wheel"]):
            scores["office"] = 0.85

        # Hospital
        if armrest and stability >= 0.5 and 450 <= seat_h <= 510:
            scores["hospital"] = 0.7

        # Wheelchair
        if leg_cfg in ["sled"] and armrest:
            scores["wheelchair"] = 0.65

        if not scores:
            return "dining"  # Default

        return max(scores, key=scores.get)

    def estimate_material(self, geometry_data: dict, chair_type: str) -> str:
        """Estimate likely material based on geometry and type."""
        vol = geometry_data.get("volume", 0)
        area = geometry_data.get("surface_area", 0)
        ratio = vol / area if area > 0 else 0

        material_map = {
            "office": "mesh-fabric" if ratio < 10 else "plastic",
            "hospital": "medical-grade plastic",
            "wheelchair": "aluminum",
            "dining": "wood" if ratio > 8 else "plastic",
            "lounge": "upholstered-foam",
            "stool": "metal",
        }
        return material_map.get(chair_type, "plastic")

    def build_feature_vector(self, geometry_data: dict, slicing_data: dict) -> list:
        """
        Build a normalized feature vector for ML/similarity tasks.
        Dimensions: [seat_h, backrest, armrest, stability, ergo, vol_norm, area_norm,
                     backrest_angle, leg_code, curvature, width, depth, height]
        """
        dims = geometry_data.get("dimensions", [500, 500, 900])
        vol = geometry_data.get("volume", 1e5)
        area = geometry_data.get("surface_area", 1e4)
        curvature = geometry_data.get("curvature_std", 0.1)

        leg_code = {"4-leg": 0.25, "pedestal": 0.5, "sled": 0.75, "5-wheel": 1.0}.get(
            slicing_data.get("leg_config", "4-leg"), 0.25
        )

        vector = [
            slicing_data.get("seat_height_mm", 0) / 1000.0,
            float(slicing_data.get("backrest_present", False)),
            float(slicing_data.get("armrest_present", False)),
            slicing_data.get("stability_score", 0),
            slicing_data.get("ergonomic_score", 0),
            min(vol / 1e7, 1.0),
            min(area / 1e6, 1.0),
            slicing_data.get("backrest_angle_deg", 0) / 90.0,
            leg_code,
            min(curvature * 10, 1.0),
            dims[0] / 1000.0 if len(dims) > 0 else 0,
            dims[1] / 1000.0 if len(dims) > 1 else 0,
            dims[2] / 1000.0 if len(dims) > 2 else 0,
        ]
        return [round(v, 5) for v in vector]

    def extract_features(self, geometry_data: dict, slicing_data: dict) -> dict:
        """Main method: extract full semantic feature set."""
        chair_type = self.classify_chair(slicing_data, geometry_data)
        material = self.estimate_material(geometry_data, chair_type)
        feature_vector = self.build_feature_vector(geometry_data, slicing_data)

        dims = geometry_data.get("dimensions", [500, 500, 900])

        features = {
            "chair_type": chair_type,
            "seat_height_mm": round(slicing_data.get("seat_height_mm", 0), 1),
            "armrests": slicing_data.get("armrest_present", False),
            "backrest": slicing_data.get("backrest_present", True),
            "backrest_angle_deg": round(slicing_data.get("backrest_angle_deg", 0), 2),
            "base_type": slicing_data.get("leg_config", "4-leg"),
            "material": material,
            "ergonomic_score": slicing_data.get("ergonomic_score", 0),
            "stability_score": slicing_data.get("stability_score", 0),
            "slice_layers": slicing_data.get("num_layers", 0),
            "width_mm": round(dims[0], 1) if len(dims) > 0 else 0,
            "depth_mm": round(dims[1], 1) if len(dims) > 1 else 0,
            "height_mm": round(dims[2], 1) if len(dims) > 2 else 0,
            "volume_cm3": round(geometry_data.get("volume", 0) / 1000, 2),
            "surface_area_cm2": round(geometry_data.get("surface_area", 0) / 100, 2),
            "triangle_count": geometry_data.get("triangle_count", 0),
            "is_watertight": geometry_data.get("is_watertight", False),
            "curvature_std": round(geometry_data.get("curvature_std", 0), 6),
            "feature_vector": feature_vector,
            "slice_signature": slicing_data.get("slice_signature", []),
            "centroid": geometry_data.get("centroid", [0, 0, 0]),
        }

        return features

    def compute_similarity(self, vec_a: list, vec_b: list) -> float:
        """Cosine similarity between two feature vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)