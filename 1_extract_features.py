import glob
import os
import json

from load import STLLoader
from slicer import CADSlicingEngine
from renderer.render import CADRenderer

from descriptors.geometry import GeometryDescriptor
from descriptors.slice_descriptor import SliceDescriptor
from descriptors.shape_dna import ShapeDNADescriptor
from descriptors.fpfh import FPFHDescriptor
from descriptors.pointnet import PointNetDescriptor
from descriptors.dinov2 import DINOv2Descriptor

# ==========================================================
# INITIALIZE COMPONENTS
# ==========================================================
loader = STLLoader()
slicer = CADSlicingEngine()
renderer = CADRenderer(num_views=12)

geom_desc = GeometryDescriptor()
slice_desc = SliceDescriptor()
shape_dna_desc = ShapeDNADescriptor(num_eigenvalues=64)
fpfh_desc = FPFHDescriptor(target_dim=33)
pointnet_desc = PointNetDescriptor(num_points=4096, target_dim=1024)
dinov2_desc = DINOv2Descriptor(target_dim=768)

os.makedirs("results", exist_ok=True)
stl_files = glob.glob("Data/*.stl")

print("========================================")
print(" DATA EXTRACTION (PRE-TRAINING PHASE)")
print("========================================")

for filepath in stl_files:
    try:
        model = loader.load_stl(filepath)
        model_id = model["model_id"]
        
        model_root_dir = os.path.join("results", model_id)
        os.makedirs(model_root_dir, exist_ok=True)

        print(f"[PROCESS] Extracting features for {model_id}...")

        # Extract Base Data
        summary = loader.get_summary(model_id)
        slice_result = slicer.slice_model(model)
        slice_result["slice_signature"] = slicer.get_slice_signature(slice_result)
        rendered_images = renderer.render_views(filepath)

        # Descriptor Extraction
        geom_features = geom_desc.extract(model)
        slice_features = slice_desc.extract(slice_result)
        shape_dna_features = shape_dna_desc.extract(model)
        fpfh_features = fpfh_desc.extract(model)
        pointnet_features = pointnet_desc.extract(model)
        dinov2_features = dinov2_desc.extract(rendered_images)
        
        # Combine all features
        all_features = {
            **geom_features, **slice_features, **shape_dna_features,
            **fpfh_features, **pointnet_features, **dinov2_features
        }
        
        # Save raw features for dataset preparation
        features_path = os.path.join(model_root_dir, "raw_features.json")
        with open(features_path, "w") as f:
            json.dump(all_features, f, indent=4)
            
        print(f"[SUCCESS] Saved raw_features.json for {model_id}\n")

    except Exception as e:
        print(f"[ERROR] Failed extracting {filepath} -> {e}\n")

print("[INFO] Data extraction complete. You can now run training/prepare_dataset.py")