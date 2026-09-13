import glob
import os
import json
import torch

from load import STLLoader
from slicer import CADSlicingEngine
from qr_encoder import LatentQREncoder
from renderer.render import CADRenderer

from descriptors.geometry import GeometryDescriptor
from descriptors.slice_descriptor import SliceDescriptor
from descriptors.shape_dna import ShapeDNADescriptor
from descriptors.fpfh import FPFHDescriptor
from descriptors.pointnet import PointNetDescriptor
from descriptors.dinov2 import DINOv2Descriptor

from descriptors.fusion import FusionMLP
from evaluation_encoding import EncodingEvaluator
from visualization_encoding import EncodingVisualizer


# ==========================================================
# Wrapper for Ablation Study Compatibility
# ==========================================================
class TrainedFusionWrapper:
    """Wraps the trained encoder so the EncodingEvaluator's ablation study can still call .fuse()"""
    def __init__(self, trained_encoder, compute_device):
        self.encoder = trained_encoder
        self.device = compute_device
        self.feature_keys = [
            "geometry_vector", "slice_vector", "shape_dna_vector",
            "fpfh_vector", "pointnet_vector", "dinov2_vector"
        ]

    def fuse(self, features: dict) -> list:
        raw_vec = []
        for key in self.feature_keys:
            raw_vec.extend(features.get(key, []))
            
        if len(raw_vec) != 1925:
            return [0.0] * 512
            
        t_input = torch.tensor(raw_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            t_latent = self.encoder(t_input)
            
        return [round(float(v), 6) for v in t_latent.squeeze(0).cpu().numpy().tolist()]


# ==========================================================
# 1. INITIALIZE PIPELINE COMPONENTS
# ==========================================================
loader = STLLoader()
slicer = CADSlicingEngine()
qr_encoder = LatentQREncoder(error_correction=1) 
renderer = CADRenderer(num_views=12)

geom_desc = GeometryDescriptor()
slice_desc = SliceDescriptor()
shape_dna_desc = ShapeDNADescriptor(num_eigenvalues=64)
fpfh_desc = FPFHDescriptor(target_dim=33)
pointnet_desc = PointNetDescriptor(num_points=4096, target_dim=1024)
dinov2_desc = DINOv2Descriptor(target_dim=768)

evaluator = EncodingEvaluator(output_dir="results/evaluation_encoding")
visualizer = EncodingVisualizer(output_dir="results")

# ==========================================================
# Load TRAINED 1925-D -> 512-D Encoder
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = FusionMLP(input_dim=1925, target_dim=512).to(device)

try:
    encoder.load_state_dict(
        torch.load("weights/encoder.pth", map_location=device, weights_only=True)
    )
    encoder.eval()
    print(f"[ENCODER] Trained encoder successfully loaded on {device}")
except Exception as e:
    print(f"[WARNING] Could not load trained encoder weights: {e}")
    print("Ensure you have run training/train_autoencoder.py first.")

# Create the wrapper for the evaluator
fusion_wrapper = TrainedFusionWrapper(encoder, device)

os.makedirs("results", exist_ok=True)
stl_files = glob.glob("Data/*.stl")

# ==========================================================
# 2. BATCH TRACKING VARIABLES
# ==========================================================
dataset_latent_vectors = []
dataset_labels = []
dataset_stl_sizes = []
dataset_qr_sizes = []
dataset_raw_features = []
dataset_pointnet_vectors = []

print("========================================")
print(" PHASE 1: CAD ENCODING & COMPRESSION")
print("========================================")

for filepath in stl_files:
    try:
        model = loader.load_stl(filepath)
        model_id = model["model_id"]
        
        folders = visualizer.create_model_folders(model_id)
        model_root_dir = os.path.join("results", model_id)

        # Extract Base Data
        summary = loader.get_summary(model_id)
        slice_result = slicer.slice_model(model)
        slice_result["slice_signature"] = slicer.get_slice_signature(slice_result)
        rendered_images = renderer.render_views(filepath)

        # Timed Descriptor Extraction
        geom_features = evaluator.measure_time("Geometry", geom_desc.extract, model)
        slice_features = evaluator.measure_time("Slice", slice_desc.extract, slice_result)
        shape_dna_features = evaluator.measure_time("ShapeDNA", shape_dna_desc.extract, model)
        fpfh_features = evaluator.measure_time("FPFH", fpfh_desc.extract, model)
        pointnet_features = evaluator.measure_time("PointNet", pointnet_desc.extract, model)
        dinov2_features = evaluator.measure_time("DINOv2", dinov2_desc.extract, rendered_images)
        
        all_features = {
            **geom_features, **slice_features, **shape_dna_features,
            **fpfh_features, **pointnet_features, **dinov2_features
        }
        dataset_raw_features.append(all_features)
        
        # Track PointNet for Dataset PCA
        if "pointnet_vector" in pointnet_features:
            dataset_pointnet_vectors.append(pointnet_features["pointnet_vector"])
        
        with open(os.path.join(model_root_dir, "raw_features.json"), "w") as f:
            json.dump(all_features, f, indent=4)

        # ==========================================================
        # FEATURE FUSION USING TRAINED ENCODER
        # ==========================================================
        raw_vector = []
        feature_keys = [
            "geometry_vector",
            "slice_vector",
            "shape_dna_vector",
            "fpfh_vector",
            "pointnet_vector",
            "dinov2_vector"
        ]

        for key in feature_keys:
            raw_vector.extend(all_features.get(key, []))

        print(f"[INFO] Fusion input dimension: {len(raw_vector)}")

        if len(raw_vector) != 1925:
            raise ValueError(f"Expected 1925 features, got {len(raw_vector)}")

        # Convert to tensor
        input_tensor = torch.tensor(raw_vector, dtype=torch.float32).unsqueeze(0).to(device)

        # Generate trained 512-D latent
        with torch.no_grad():
            latent_tensor = encoder(input_tensor)

        latent_list = latent_tensor.squeeze(0).cpu().numpy().tolist()
        latent_vector = [round(float(v), 6) for v in latent_list]

        print(f"[SUCCESS] Generated trained {len(latent_vector)}-D latent vector")

        # ==========================================================
        # Generate Per-Model Thesis Visualizations
        # ==========================================================
        visualizer.plot_input_statistics(summary, folders)
        visualizer.plot_slice_analysis(slice_result, folders)
        visualizer.plot_geometry_features(geom_features, folders)
        visualizer.plot_shapedna(shape_dna_features, folders)
        visualizer.plot_fpfh(fpfh_features, folders)
        visualizer.plot_pointnet(pointnet_features, model.get("trimesh"), folders)
        visualizer.plot_dinov2(dinov2_features, rendered_images, folders)
        visualizer.plot_fusion(all_features, latent_vector, folders)

        # ==========================================================
        # Generate QR Code
        # ==========================================================
        qr_path_main = os.path.join(model_root_dir, "latent_qr.png")
        qr_path_viz = os.path.join(folders["qr"], "qr_code.png")
        
        payload = {"latent_vector": latent_vector}
        
        def encode_and_save():
            encoded = qr_encoder.compress_features(payload)
            qr_encoder.generate_qr(encoded, qr_path_main)
            qr_encoder.generate_qr(encoded, qr_path_viz) 
            return encoded
            
        encoded_payload = evaluator.measure_time("QREncoding", encode_and_save)

        # Tracking for Batch Visualizations
        dataset_latent_vectors.append(latent_vector)
        dataset_labels.append(model_id)
        
        stl_size = os.path.getsize(filepath)
        qr_payload_size = len(encoded_payload.encode('utf-8'))
        dataset_stl_sizes.append(stl_size)
        dataset_qr_sizes.append(qr_payload_size)
        
        evaluator.record_compression(model_id, stl_size, qr_payload_size)
        print(f"[SUCCESS] Encoded and visualized {model_id}\n")

    except Exception as e:
        print(f"[ERROR] Failed encoding {filepath} -> {e}\n")


# ==========================================================
# 3. DATASET-LEVEL VISUALIZATION & REPORTING
# ==========================================================
print("========================================")
print(" GENERATING DATASET-LEVEL REPORTS")
print("========================================")
dataset_visualizations_dir = "results/dataset_visualizations"

if dataset_latent_vectors:
    visualizer.plot_dataset_tsne(dataset_latent_vectors, dataset_labels, dataset_visualizations_dir)
    visualizer.plot_dataset_pca(dataset_latent_vectors, dataset_labels, dataset_visualizations_dir)
    
    if dataset_pointnet_vectors:
        visualizer.plot_dataset_pointnet_pca(dataset_pointnet_vectors, dataset_labels, dataset_visualizations_dir)
    
    if hasattr(visualizer, 'plot_similarity_matrix'):
        visualizer.plot_similarity_matrix(dataset_latent_vectors, dataset_labels)
    if hasattr(visualizer, 'plot_compression_stats'):
        visualizer.plot_compression_stats(dataset_stl_sizes, dataset_qr_sizes, dataset_labels)
    
    # Passing the wrapper here keeps the ablation study functional
    evaluator.generate_report(dataset_latent_vectors, dataset_labels, dataset_raw_features, fusion_wrapper)
else:
    print("[WARNING] No models were successfully processed. Skipping batch visualizations.")