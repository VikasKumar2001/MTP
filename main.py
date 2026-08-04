import glob
import os
import json
from load import STLLoader
from slicer import CADSlicingEngine
from feature import FeatureIntelligenceSystem
from qr_encoder import LatentQREncoder  # Imported the new QR encoder class

# Initialize paths and engine components
stl_files = glob.glob("Data/*.stl")

loader = STLLoader()
slicer = CADSlicingEngine()
feature_system = FeatureIntelligenceSystem()
qr_encoder = LatentQREncoder()  # Instantiated the encoder

os.makedirs("results", exist_ok=True)

for filepath in stl_files:
    try:
        # 1. Load Model
        model = loader.load_stl(filepath)
        model_id = model["model_id"]

        # Setup output directory and file paths
        save_folder = os.path.join("results", model_id)
        graph_path = os.path.join(save_folder, "slice_graph.png")
        qr_path = os.path.join(save_folder, "latent_qr.png")  # Path for the new QR code

        os.makedirs(save_folder, exist_ok=True)

        # 2. Get Geometry Summary
        summary = loader.get_summary(model_id)
        with open(os.path.join(save_folder, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        # 3. Slice Model & Generate Signatures
        slice_result = slicer.slice_model(model)
        slice_result["slice_signature"] = slicer.get_slice_signature(slice_result)
        
        with open(os.path.join(save_folder, "slice_result.json"), "w") as f:
            json.dump(slice_result, f, indent=4)
            slicer.plot_slice_graph(slice_result, graph_path)

        # 4. Extract Semantic Features
        features = feature_system.extract_features(
            geometry_data=summary,
            slicing_data=slice_result
        )

        with open(os.path.join(save_folder, "features.json"), "w") as f:
            json.dump(features, f, indent=4)

        # 5. Compress Features and Generate QR Code
        encoded_payload = qr_encoder.compress_features(features)
        qr_encoder.generate_qr(encoded_payload, qr_path)

        print(f"[SUCCESS] Saved data and generated QR code for {model_id}")

    except Exception as e:
        print(f"[ERROR] {filepath} -> {e}")