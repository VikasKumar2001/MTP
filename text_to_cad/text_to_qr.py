import os
import sys
import numpy as np
import torch

from text_to_cad.ollama_client import get_text_embedding
from text_to_cad.text_adapter import TextToCADAdapter
from qr_encoder import LatentQREncoder

# Project setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "weights", "text_to_cad.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "text_to_cad", "query_qr")
OUTPUT_QR = os.path.join(OUTPUT_DIR, "query_qr.png")
CAD_DIM = 512

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained text → CAD adapter
if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(
        f"Text-to-CAD adapter not found: {ADAPTER_PATH}\nRun train_text_adapter.py first."
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

for param in adapter.parameters():
    param.requires_grad = False

print("=" * 70)
print("TEXT → CAD → QR")
print("=" * 70)
print(f"[INFO] Device: {device}")
print(f"[INFO] Text dimension: {text_dim}")
print(f"[INFO] CAD dimension: {cad_dim}")

# Get query
query = input("\nEnter your CAD search query:\n> ").strip()
if not query:
    raise ValueError("Search query cannot be empty.")

print("\n[QUERY]")
print(query)

# Generate Ollama text embedding
print("\n[1/3] Generating Ollama embedding...")
text_embedding = get_text_embedding(query)
text_embedding = np.asarray(text_embedding, dtype=np.float32)

if text_embedding.shape != (text_dim,):
    raise ValueError(
        f"Ollama returned dimension {text_embedding.shape[0]}, "
        f"but adapter expects {text_dim}."
    )

text_tensor = torch.tensor(text_embedding, dtype=torch.float32).unsqueeze(0).to(device)

# Map text to CAD latent space
print("[2/3] Mapping text into CAD latent space...")
with torch.no_grad():
    cad_latent = adapter(text_tensor)

cad_latent = cad_latent.squeeze(0).cpu().numpy().astype(np.float32)

if cad_latent.shape != (CAD_DIM,):
    raise ValueError(f"Expected {CAD_DIM}-D vector, got {cad_latent.shape}")

print(f"[SUCCESS] Generated {CAD_DIM}-D CAD query vector")
print(f"[INFO] L2 norm: {np.linalg.norm(cad_latent):.6f}")

# Generate QR code
print("[3/3] Generating query QR...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

latent_vector = [round(float(v), 6) for v in cad_latent.tolist()]
payload = {"latent_vector": latent_vector}

qr_encoder = LatentQREncoder(error_correction=1)
encoded_payload = qr_encoder.compress_features(payload)
qr_encoder.generate_qr(encoded_payload, OUTPUT_QR)

# Complete
print("\n" + "=" * 70)
print("QUERY QR GENERATED")
print("=" * 70)
print(f"Query:\n  {query}")
print(f"\nQR:\n  {OUTPUT_QR}")

print(
    f"\nVector dimension: "
    f"{len(latent_vector)}"
)

print(
    "\nNext step:"
)

print(
    "Use this query QR with your existing "
    "QR similarity engine."
)

print("=" * 70)