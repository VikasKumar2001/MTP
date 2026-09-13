import os
import time
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SystemEvaluator:

    def __init__(self, output_dir="results/evaluation"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.timings = {
            "Geometry": [], "Slice": [], "ShapeDNA": [], 
            "FPFH": [], "PointNet": [], "DINOv2": [], "QREncoding": []
        }
        self.compression_data = []

    def measure_time(self, module_name: str, func, *args, **kwargs):

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        if module_name in self.timings:
            self.timings[module_name].append(elapsed)
            
        return result

    def record_compression(self, model_id: str, stl_bytes: int, qr_bytes: int):
        """Calculates and stores the compression ratio."""
        ratio = stl_bytes / qr_bytes if qr_bytes > 0 else 0
        self.compression_data.append({
            "model_id": model_id,
            "stl_size_bytes": stl_bytes,
            "qr_payload_bytes": qr_bytes,
            "compression_ratio": ratio
        })

    def evaluate_retrieval(self, latent_vectors: list, labels: list) -> dict:
      
        if len(latent_vectors) < 2:
            return {"top_1_accuracy": 0.0, "top_5_accuracy": 0.0}

        sim_matrix = cosine_similarity(np.array(latent_vectors))
        
        # Extract ground truth classes by splitting the label (e.g., 'chair_01' -> 'chair')
        classes = [label.split('_')[0] if '_' in label else label for label in labels]
        
        top_1_correct = 0
        top_5_correct = 0
        
        for i in range(len(latent_vectors)):
            query_class = classes[i]
            
            # Sort indices by similarity (descending), excluding the query itself (index 0)
            sorted_indices = np.argsort(sim_matrix[i])[::-1][1:]
            
            top_1_class = classes[sorted_indices[0]]
            top_5_classes = [classes[idx] for idx in sorted_indices[:5]]
            
            if top_1_class == query_class:
                top_1_correct += 1
            if query_class in top_5_classes:
                top_5_correct += 1
                
        n_samples = len(latent_vectors)
        return {
            "top_1_accuracy": round((top_1_correct / n_samples) * 100, 2),
            "top_5_accuracy": round((top_5_correct / n_samples) * 100, 2)
        }

    def run_ablation_study(self, base_features_list: list, labels: list, fusion_engine) -> dict:
      
        ablation_results = {}
        modalities = ["fpfh_vector", "slice_vector", "pointnet_vector", "dinov2_vector", "shape_dna_vector"]
        
        for modality_to_drop in modalities:
            ablated_latents = []
            
            for features in base_features_list:
                # Create a copy of the features and zero out the dropped modality
                ablated_features = features.copy()
                if modality_to_drop in ablated_features:
                    dim_length = len(ablated_features[modality_to_drop])
                    ablated_features[modality_to_drop] = [0.0] * dim_length
                
                # Fuse the ablated feature set
                latent = fusion_engine.fuse(ablated_features)
                ablated_latents.append(latent)
                
            # Compute accuracy without this modality
            metrics = self.evaluate_retrieval(ablated_latents, labels)
            ablation_results[f"Without {modality_to_drop.split('_')[0].upper()}"] = metrics["top_1_accuracy"]
            
        return ablation_results

    def generate_report(self, latent_vectors: list, labels: list, base_features_list: list, fusion_engine):
        print("\n[INFO] Generating Evaluation Report...")
        
        # 1. Average Timings
        avg_timings = {k: round(np.mean(v), 4) if v else 0.0 for k, v in self.timings.items()}
        
        # 2. Average Compression
        avg_ratio = np.mean([d["compression_ratio"] for d in self.compression_data]) if self.compression_data else 0.0
        
        # 3. Baseline Retrieval Accuracy
        retrieval_metrics = self.evaluate_retrieval(latent_vectors, labels)
        
        # 4. Ablation Study
        ablation_metrics = self.run_ablation_study(base_features_list, labels, fusion_engine)
        
        report = {
            "Average_Execution_Times_Seconds": avg_timings,
            "Compression": {
                "Average_Ratio": round(avg_ratio, 2),
                "Details": self.compression_data
            },
            "Baseline_Retrieval_Accuracy": retrieval_metrics,
            "Ablation_Study_Top1_Accuracy": ablation_metrics
        }
        
        report_path = os.path.join(self.output_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        print(f"[SUCCESS] Evaluation report saved to {report_path}")
        
        # Print summary to console
        print("\n--- PERFORMANCE SUMMARY ---")
        print(f"Top-1 Retrieval Accuracy: {retrieval_metrics['top_1_accuracy']}%")
        print(f"Top-5 Retrieval Accuracy: {retrieval_metrics['top_5_accuracy']}%")
        print(f"Average Compression Ratio: {round(avg_ratio, 1)}x")
        print("---------------------------\n")