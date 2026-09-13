import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
ROOT_DIR = os.path.dirname(

    os.path.dirname(

        os.path.abspath(__file__)

    )

)



sys.path.insert(0, ROOT_DIR)
from sklearn.metrics.pairwise import cosine_similarity

# Assuming qr_decoder.py is in the same root directory
from qr_decoder import LatentQRDecoder


class QRSimilarityEngine:
    """
    CAD Retrieval System

    Workflow:
        Query QR -> QR Decoding -> 512-D Latent Vector -> 
        Compare with all stored QR latent vectors -> 
        Cosine Similarity -> Ranked CAD Retrieval
    """
    def __init__(self, qr_directory="all_qr", output_directory="results/similarity_search"):
        self.qr_directory = qr_directory
        self.output_directory = output_directory

        # Create output directory
        os.makedirs(self.output_directory, exist_ok=True)

        # Initialize existing decoder
        self.decoder = LatentQRDecoder(expected_dim=512)

    # ==========================================================
    # Decode QR → 512-D latent vector
    # ==========================================================
    def get_latent_from_qr(self, qr_path: str) -> list:
        latent_vector = self.decoder.decode(qr_path)

        # Validate decoded vector (decoder returns zeros on failure)
        if (
            latent_vector is None
            or len(latent_vector) == 0
            or sum(abs(float(v)) for v in latent_vector) == 0
        ):
            raise ValueError(f"Decoded vector is empty or invalid: {qr_path}")

        # Ensure correct dimension
        if len(latent_vector) != 512:
            raise ValueError(f"Expected 512-D vector but got {len(latent_vector)}-D: {qr_path}")

        return latent_vector

    # ==========================================================
    # Cosine similarity
    # ==========================================================
    def calculate_similarity(self, vector1: list, vector2: list) -> float:
        vec1 = np.asarray(vector1, dtype=np.float32).reshape(1, -1)
        vec2 = np.asarray(vector2, dtype=np.float32).reshape(1, -1)

        similarity = cosine_similarity(vec1, vec2)[0][0]
        return float(similarity)

    # ==========================================================
    # Find all QR files
    # ==========================================================
    def find_qr_files(self) -> list:
        qr_files = []

        if not os.path.exists(self.qr_directory):
            print(f"[ERROR] QR directory does not exist: {self.qr_directory}")
            return qr_files

        for root, dirs, files in os.walk(self.qr_directory):
            for file in files:
                # Assumes your flat folder structure containing only QR pngs
                if file.lower().endswith(".png"):
                    full_path = os.path.join(root, file)
                    qr_files.append(full_path)

        return qr_files

    # ==========================================================
    # Extract model ID from QR filename
    # ==========================================================
    def get_model_id(self, qr_path: str) -> str:
        # Example: all_qr/chair_001.png -> chair_001
        filename = os.path.basename(qr_path)
        model_id = os.path.splitext(filename)[0]
        return model_id

    # ==========================================================
    # Similarity Search
    # ==========================================================
    def search(self, query_qr_path: str) -> list:
        print("=" * 75)
        print("              LATENT QR CAD SIMILARITY SEARCH")
        print("=" * 75)
        print(f"\n[INFO] Query QR: {query_qr_path}")

        # Decode query QR
        try:
            query_vector = self.get_latent_from_qr(query_qr_path)
        except Exception as e:
            print(f"[ERROR] Failed to decode query QR:\n{e}")
            return []

        print(f"[INFO] Query latent dimension: {len(query_vector)}")

        # Find candidate QR codes
        qr_files = self.find_qr_files()
        print(f"[INFO] Found {len(qr_files)} candidate QR codes.")

        if not qr_files:
            print("[ERROR] No QR files found in database.")
            return []

        # Compare query against candidates
        results = []
        query_absolute = os.path.abspath(query_qr_path)

        for qr_path in qr_files:
            # Don't compare QR with its exact same file path
            if os.path.abspath(qr_path) == query_absolute:
                continue

            try:
                candidate_vector = self.get_latent_from_qr(qr_path)

                # Dimension check
                if len(candidate_vector) != len(query_vector):
                    print(f"[SKIP] Dimension mismatch: {qr_path}")
                    continue

                # Calculate similarity
                similarity = self.calculate_similarity(query_vector, candidate_vector)
                model_id = self.get_model_id(qr_path)

                results.append({
                    "model_id": model_id,
                    "qr_path": qr_path,
                    "similarity": similarity
                })

            except Exception as e:
                print(f"[ERROR] Skipping {qr_path}: {e}")

        # Sort highest similarity first
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    # ==========================================================
    # Similarity Category
    # ==========================================================
    def similarity_category(self, score: float) -> str:
        if score >= 0.90: return "Very High"
        elif score >= 0.75: return "High"
        elif score >= 0.60: return "Moderate"
        elif score >= 0.40: return "Low"
        else: return "Very Low"

    # ==========================================================
    # Print Results
    # ==========================================================
    def print_results(self, results: list, top_k: int = 10):
        if not results:
            print("\nNo matching results found.")
            return

        print("\n")
        print(f"{'Rank':<7}{'Model ID':<25}{'Similarity':<15}{'Category'}")
        print("-" * 75)

        for i, result in enumerate(results[:top_k], start=1):
            score = result["similarity"]
            percentage = score * 100
            category = self.similarity_category(score)

            print(f"{i:<7}{result['model_id']:<25}{percentage:>7.2f}%       {category}")
        print("-" * 75)

    # ==========================================================
    # Save CSV
    # ==========================================================
    def save_csv(self, results: list):
        if not results: return
        
        csv_path = os.path.join(self.output_directory, "similarity_results.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Rank", "Model ID", "QR Path", 
                "Cosine Similarity", "Similarity Percentage", "Category"
            ])

            for i, result in enumerate(results, start=1):
                score = result["similarity"]
                writer.writerow([
                    i, 
                    result["model_id"], 
                    result["qr_path"],
                    round(score, 6), 
                    round(score * 100, 2),
                    self.similarity_category(score)
                ])

        print(f"\n[INFO] CSV saved to:\n{csv_path}")

    # ==========================================================
    # Plot Similarity Results
    # ==========================================================
    def plot_similarity(self, results: list, top_k: int = 10):
        if not results: return

        top_results = results[:top_k]
        model_ids = [r["model_id"] for r in top_results]
        similarities = [r["similarity"] * 100 for r in top_results]

        # Reverse so highest is displayed at top of the bar chart
        model_ids = model_ids[::-1]
        similarities = similarities[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(model_ids, similarities, color='steelblue')
        plt.xlabel("Cosine Similarity (%)")
        plt.ylabel("CAD Model")
        plt.title("Top CAD Models Retrieved by Latent QR Similarity")
        plt.xlim(0, 100)
        plt.grid(axis="x", alpha=0.3)

        # Add values to bars
        for i, value in enumerate(similarities):
            plt.text(value + 1, i, f"{value:.2f}%", va="center")

        plt.tight_layout()
        plot_path = os.path.join(self.output_directory, "similarity_ranking.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Similarity graph saved to:\n{plot_path}")

    # ==========================================================
    # Save Top Match
    # ==========================================================
    def print_best_match(self, results: list):
        if not results: return
        best = results[0]

        print("\n")
        print("=" * 75)
        print("                    BEST MATCH")
        print("=" * 75)
        print(f"Model ID     : {best['model_id']}")
        print(f"QR Path      : {best['qr_path']}")
        print(f"Similarity   : {best['similarity'] * 100:.2f}%")
        print(f"Category     : {self.similarity_category(best['similarity'])}")
        print("=" * 75)


if __name__ == "__main__":
    
    QR_DATABASE = "all_qr"
    QUERY_QR = "C:\\Users\\kvika\\Desktop\\MTP\\Prompt_to_QR\\text_to_cad\\query_qr\\query_qr.png"
    TOP_K = 10
    
    # Execution Safety Checks
    if not os.path.exists(QUERY_QR):
        print(f"[ERROR] Query file '{QUERY_QR}' not found. Please provide a valid QR image.")
        exit(1)
        
    if not os.path.exists(QR_DATABASE):
        print(f"[ERROR] Database directory '{QR_DATABASE}' not found. Please run your encoding pipeline first.")
        exit(1)

    engine = QRSimilarityEngine(
        qr_directory=QR_DATABASE,
        output_directory="results/similarity_search"
    )

    results = engine.search(QUERY_QR)

    if results:
        engine.print_results(results, top_k=TOP_K)
        engine.print_best_match(results)
        engine.save_csv(results)
        engine.plot_similarity(results, top_k=TOP_K)