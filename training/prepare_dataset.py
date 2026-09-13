import os
import glob
import json
import torch
import numpy as np
import trimesh
from torch.utils.data import Dataset

# ==========================================================
# Project Root
# ==========================================================
# prepare_dataset.py is inside: New_Try/training/
# Therefore:
# dirname(__file__)          -> training
# dirname(dirname(__file__)) -> New_Try

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ==========================================================
# Dataset
# ==========================================================
class CADDataset(Dataset):
    """
    Custom PyTorch Dataset for 3D CAD Reconstruction.
    """
    def __init__(self, inputs_tensor, targets_tensor):
        self.inputs = inputs_tensor
        self.targets = targets_tensor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ==========================================================
# Dataset Preparation
# ==========================================================
def build_and_save_dataset(data_dir=None, results_dir=None, save_path=None, num_samples=642):
    print("=" * 60)
    print("              CAD DATASET PREPARATION")
    print("=" * 60)

    # ------------------------------------------------------
    # Correct project paths
    # ------------------------------------------------------
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "Data")
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "results")
    if save_path is None:
        save_path = os.path.join(SCRIPT_DIR, "dataset.pt")

    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] STL directory: {data_dir}")
    print(f"[INFO] Results dir  : {results_dir}")
    print(f"[INFO] Dataset path : {save_path}\n")

    # ------------------------------------------------------
    # Check directories
    # ------------------------------------------------------
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory does not exist:\n{data_dir}")
        return

    if not os.path.exists(results_dir):
        print(f"[ERROR] Results directory does not exist:\n{results_dir}")
        return

    # ------------------------------------------------------
    # Find STL files
    # ------------------------------------------------------
    stl_files = glob.glob(os.path.join(data_dir, "*.stl"))
    print(f"[INFO] Found {len(stl_files)} STL files.")

    if len(stl_files) == 0:
        print("[ERROR] No STL files found.")
        return

    # ------------------------------------------------------
    # Storage
    # ------------------------------------------------------
    inputs = []
    targets = []
    successful = 0
    skipped = 0

    # ------------------------------------------------------
    # Process each STL
    # ------------------------------------------------------
    for filepath in stl_files:
        model_id = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\n[PROCESSING] {model_id}")

        # Raw feature path
        features_path = os.path.join(results_dir, model_id, "raw_features.json")

        if not os.path.exists(features_path):
            print(f"[SKIP] raw_features.json not found:\n       {features_path}")
            skipped += 1
            continue

        try:
            # ==================================================
            # 1. Load STL
            # ==================================================
            mesh = trimesh.load(filepath, force="mesh")
            
            if mesh is None:
                raise ValueError("Could not load mesh.")

            # Normalize mesh
            mesh.vertices -= mesh.centroid
            max_extents = mesh.extents.max()
            
            if max_extents > 0:
                mesh.vertices /= max_extents

            # ==================================================
            # 2. Sample target points
            # ==================================================
            pts, _ = trimesh.sample.sample_surface(mesh, num_samples)
            targets.append(pts)

            # ==================================================
            # 3. Load raw features
            # ==================================================
            with open(features_path, "r", encoding="utf-8") as f:
                features_dict = json.load(f)

            # ==================================================
            # 4. Concatenate features
            # IMPORTANT: This order MUST remain identical to the 
            # order used during training and encoding.
            # ==================================================
            flat_vector = []
            feature_keys = [
                "geometry_vector",
                "slice_vector",
                "shape_dna_vector",
                "fpfh_vector",
                "pointnet_vector",
                "dinov2_vector"
            ]

            for key in feature_keys:
                feature = features_dict.get(key, [])
                flat_vector.extend(feature)

            # Check feature dimension
            print(f"[INFO] Feature dimension: {len(flat_vector)}")

            if len(flat_vector) != 1925:
                print(f"[WARNING] Expected 1925 features but got {len(flat_vector)}")
                # Do not include invalid samples; remove the points we just appended
                targets.pop()
                skipped += 1
                continue

            # ==================================================
            # 5. Store
            # ==================================================
            inputs.append(flat_vector)
            successful += 1
            print("[SUCCESS] Model processed.")

        except Exception as e:
            print(f"[ERROR] Failed processing {model_id}: {e}")
            # If it failed after appending to targets but before appending to inputs, align them
            if len(targets) > len(inputs):
                targets.pop()
            skipped += 1

    # ==========================================================
    # Check dataset
    # ==========================================================
    print("\n" + "=" * 60)
    print("                DATASET SUMMARY")
    print("=" * 60)
    print(f"Successful models : {successful}")
    print(f"Skipped models    : {skipped}")

    if len(inputs) == 0:
        print("[ERROR] No valid training samples found.")
        return

    # ==========================================================
    # Convert to tensors
    # ==========================================================
    inputs_array = np.asarray(inputs, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)

    print(f"\n[INFO] Input shape : {inputs_array.shape}")
    print(f"[INFO] Target shape: {targets_array.shape}")

    # Verify dimensions
    if inputs_array.shape[1] != 1925:
        raise ValueError(f"Expected input dimension 1925, got {inputs_array.shape[1]}")

    # ==========================================================
    # Convert to PyTorch
    # ==========================================================

    inputs_tensor = torch.tensor(
        inputs_array,
        dtype=torch.float32
    )

    targets_tensor = torch.tensor(
        targets_array,
        dtype=torch.float32
    )

    # ==========================================================
    # Save plain tensors
    # ==========================================================

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    torch.save(
        {
            "inputs": inputs_tensor,
            "targets": targets_tensor
        },
        save_path
    )

    print()
    print("=" * 60)
    print("[SUCCESS] Dataset saved!")
    print(f"[INFO] Path   : {save_path}")
    print(f"[INFO] Samples: {len(inputs_tensor)}")
    print(f"[INFO] Input  : {inputs_tensor.shape}")
    print(f"[INFO] Target : {targets_tensor.shape}")
    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    build_and_save_dataset()