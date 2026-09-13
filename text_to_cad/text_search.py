import os
import sys
import numpy as np
import torch

from text_to_cad.ollama_client import get_text_embedding
from text_to_cad.text_adapter import TextToCADAdapter
from qr_encoder import LatentQREncoder
from QR_similarity.qr_similarity import QRSimilarityEngine

# Project setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "weights", "text_to_cad.pth")
QR_DATABASE = os.path.join(PROJECT_ROOT, "QR_similarity", "all_qr")
QUERY_QR_DIR = os.path.join(PROJECT_ROOT, "text_to_cad", "query_qr")
QUERY_QR_PATH = os.path.join(QUERY_QR_DIR, "query_qr.png")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "text_search")

# Configuration
CAD_DIM = 512
TOP_K = 10

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load text → CAD adapter
if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(
        f"Adapter not found: {ADAPTER_PATH}\nRun train_text_adapter.py first."
    )

checkpoint = torch.load(ADAPTER_PATH, map_location=device, weights_only=True)
text_dim = checkpoint["text_dim"]
cad_dim = checkpoint["cad_dim"]

if cad_dim != CAD_DIM:
    raise ValueError(
        f"Expected CAD dimension {CAD_DIM}, but checkpoint contains {cad_dim}"
    )

adapter = TextToCADAdapter(text_dim=text_dim, cad_dim=cad_dim).to(device)
adapter.load_state_dict(checkpoint["model_state_dict"])
adapter.eval()

for parameter in adapter.parameters():
    parameter.requires_grad = False

def text_to_cad_latent(query):
    """Convert text query to 512-D CAD latent vector."""
    print("\n[1/4] Generating Ollama embedding...")
    embedding = get_text_embedding(query)
    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.shape != (text_dim,):
        raise ValueError(
            f"Ollama returned {embedding.shape[0]} dimensions, "
            f"but adapter expects {text_dim}."
        )

    text_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    print("[2/4] Mapping text into 512-D CAD space...")

    with torch.no_grad():
        latent_tensor = adapter(text_tensor)

    latent = latent_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    if latent.shape != (CAD_DIM,):
        raise ValueError(f"Expected {CAD_DIM}-D latent, got {latent.shape}")

    return latent


def generate_query_qr(latent):
    """Generate QR code from CAD latent vector."""
    print("[3/4] Generating query QR...")
    os.makedirs(QUERY_QR_DIR, exist_ok=True)

    latent_vector = [round(float(v), 6) for v in latent.tolist()]
    payload = {"latent_vector": latent_vector}

    qr_encoder = LatentQREncoder(error_correction=1)
    encoded_payload = qr_encoder.compress_features(payload)
    qr_encoder.generate_qr(encoded_payload, QUERY_QR_PATH)

    print(f"[SUCCESS] Query QR:\n{QUERY_QR_PATH}")


def search_cad(query):
    """Search CAD database using natural language query."""
    print("\n" + "=" * 75)
    print("              QRFusionCAD STAGE 2")
    print("          NATURAL LANGUAGE CAD SEARCH")
    print("=" * 75)
    print(f"\nQuery: {query}\n")

    # Text → CAD
    latent = text_to_cad_latent(query)
    print(f"[SUCCESS] Generated {len(latent)}-D CAD latent.")

    # CAD latent → QR
    generate_query_qr(latent)

    # QR similarity search
    print("\n[4/4] Searching CAD QR database...")
    if not os.path.exists(QR_DATABASE):
        raise FileNotFoundError(
            f"CAD QR database not found: {QR_DATABASE}\nRun merger.py first."
        )

    engine = QRSimilarityEngine(qr_directory=QR_DATABASE, output_directory=OUTPUT_DIR)
    results = engine.search(QUERY_QR_PATH)

    # Display results
    if not results:
        print("\nNo matching CAD models found.")
        return []

    print("\n" + "=" * 75)
    print("                    SEARCH RESULTS")
    print("=" * 75)
    print(
        f"{'Rank':<8}{'CAD Model':<35}{'Similarity':<15}Category"
    )
    print("-" * 75)

    for i, result in enumerate(results[:TOP_K], start=1):
        score = result["similarity"]
        percentage = score * 100
        category = engine.similarity_category(score)
        print(
            f"{i:<8}{result['model_id']:<35}{percentage:>7.2f}%       {category}"
        )

    print("-" * 75)

    # Best match
    best = results[0]
    print("\n" + "=" * 75)
    print("                    BEST MATCH")
    print("=" * 75)
    print(f"Model ID   : {best['model_id']}")
    print(f"Similarity : {best['similarity'] * 100:.2f}%")
    print(f"Category   : {engine.similarity_category(best['similarity'])}")
    print(f"QR         : {best['qr_path']}")
    print("=" * 75)

    # Save results
    engine.save_csv(results)
    return results



if __name__ == "__main__":
    query = input("\nEnter your CAD search query:\n> ").strip()

    if not query:
        print("[ERROR] Query cannot be empty.")
        sys.exit(1)

    try:
        search_cad(query)
    except Exception as e:
        print("\n[ERROR]")
        print(e)