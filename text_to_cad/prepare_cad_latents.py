import glob
import os
import numpy as np
import torch
import sys

from load import STLLoader
from slicer import CADSlicingEngine
from renderer.render import CADRenderer

from descriptors.geometry import GeometryDescriptor
from descriptors.slice_descriptor import SliceDescriptor
from descriptors.shape_dna import ShapeDNADescriptor
from descriptors.fpfh import FPFHDescriptor
from descriptors.pointnet import PointNetDescriptor
from descriptors.dinov2 import DINOv2Descriptor

from descriptors.fusion import FusionMLP

# Project setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "encoder.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "text_to_cad", "cad_latents")

# Configuration
INPUT_DIM = 1925
LATENT_DIM = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print setup
print("=" * 70)
print("PREPARING CAD LATENT VECTORS")
print("=" * 70)
print(f"[INFO] Device: {device}")
print(f"[INFO] Encoder: {WEIGHTS_PATH}")

# Load frozen CAD encoder
encoder = FusionMLP(input_dim=INPUT_DIM, target_dim=LATENT_DIM).to(device)
encoder.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
encoder.eval()

# Freeze encoder completely
for param in encoder.parameters():
    param.requires_grad = False

print("[SUCCESS] Frozen CAD encoder loaded.")

# Initialize existing pipeline
loader = STLLoader()
slicer = CADSlicingEngine()
renderer = CADRenderer(num_views=12)

geom_desc = GeometryDescriptor()
slice_desc = SliceDescriptor()
shape_dna_desc = ShapeDNADescriptor(num_eigenvalues=64)
fpfh_desc = FPFHDescriptor(target_dim=33)
pointnet_desc = PointNetDescriptor(num_points=4096, target_dim=1024)
dinov2_desc = DINOv2Descriptor(target_dim=768)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find STL files
stl_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.stl")))
print(f"[INFO] Found {len(stl_files)} STL files.")

# Feature order
feature_keys = [
    "geometry_vector",
    "slice_vector",
    "shape_dna_vector",
    "fpfh_vector",
    "pointnet_vector",
    "dinov2_vector",
]

# Process each STL
for index, filepath in enumerate(stl_files):
    filename = os.path.basename(filepath)

    print("\n" + "=" * 70)
    print(f"[{index + 1}/{len(stl_files)}] {filename}")
    print("=" * 70)

    try:
        # Load STL
        model = loader.load_stl(filepath)
        model_id = model["model_id"]
        print(f"[INFO] Model ID: {model_id}")

        # Base CAD information
        slice_result = slicer.slice_model(model)
        slice_result["slice_signature"] = slicer.get_slice_signature(slice_result)

        # Render views for DINOv2
        rendered_images = renderer.render_views(filepath)

        # Extract six descriptors
        geom_features = geom_desc.extract(model)
        slice_features = slice_desc.extract(slice_result)
        shape_dna_features = shape_dna_desc.extract(model)
        fpfh_features = fpfh_desc.extract(model)
        pointnet_features = pointnet_desc.extract(model)
        dinov2_features = dinov2_desc.extract(rendered_images)

        # Combine exactly like main_encode.py
        all_features = {
            **geom_features,
            **slice_features,
            **shape_dna_features,
            **fpfh_features,
            **pointnet_features,
            **dinov2_features,
        }

        raw_vector = []
        for key in feature_keys:
            raw_vector.extend(all_features.get(key, []))


        print(
            f"[INFO] Fusion input dimension: "
            f"{len(raw_vector)}"
        )


        # ----------------------------------------------------
        # Verify 1925-D
        # ----------------------------------------------------

        if len(raw_vector) != INPUT_DIM:

            raise ValueError(
                f"Expected {INPUT_DIM} features, "
                f"got {len(raw_vector)}"
            )


        # ----------------------------------------------------
        # Convert to tensor
        # ----------------------------------------------------

        input_tensor = torch.tensor(
            raw_vector,
            dtype=torch.float32
        ).unsqueeze(0).to(device)


        # ----------------------------------------------------
        # Generate frozen 512-D CAD latent
        # ----------------------------------------------------

        with torch.no_grad():

            latent_tensor = encoder(
                input_tensor
            )


        latent = (
            latent_tensor
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


        # ----------------------------------------------------
        # Verify latent dimension
        # ----------------------------------------------------

        if latent.shape != (LATENT_DIM,):

            raise ValueError(
                f"Expected latent shape "
                f"({LATENT_DIM},), "
                f"got {latent.shape}"
            )


        # ----------------------------------------------------
        # Save .npy
        # ----------------------------------------------------

        output_filename = (
            os.path.splitext(filename)[0]
            + ".npy"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_filename
        )

        np.save(
            output_path,
            latent
        )


        print(
            f"[SUCCESS] Saved: {output_path}"
        )

        print(
            f"[INFO] Latent dimension: "
            f"{latent.shape[0]}"
        )


    except Exception as e:

        print(
            f"[ERROR] Failed processing "
            f"{filename}: {e}"
        )

# Complete
print("\n" + "=" * 70)
print("CAD LATENT PREPARATION COMPLETE")
print("=" * 70)
print(f"[INFO] Latents saved in:\n{OUTPUT_DIR}")