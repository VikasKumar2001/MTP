import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

from ollama_client import get_text_embedding
from text_adapter import TextToCADAdapter

# Project setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "weights", "text_to_cad.pth")
CAD_LATENT_DIR = os.path.join(PROJECT_ROOT, "text_to_cad", "cad_latents")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load adapter
checkpoint = torch.load(ADAPTER_PATH, map_location=device, weights_only=True)
text_dim = checkpoint["text_dim"]
cad_dim = checkpoint["cad_dim"]

adapter = TextToCADAdapter(text_dim=text_dim, cad_dim=cad_dim).to(device)
adapter.load_state_dict(checkpoint["model_state_dict"])
adapter.eval()


# Load CAD latents
cad_latents = {}
for filename in os.listdir(CAD_LATENT_DIR):
    if not filename.endswith(".npy"):
        continue

    path = os.path.join(CAD_LATENT_DIR, filename)
    vector = np.load(path).astype(np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    cad_latents[os.path.splitext(filename)[0]] = vector


# Get query from user
query = input("\nEnter CAD search query:\n> ").strip()
if not query:
    raise ValueError("Query cannot be empty.")

print("\n" + "=" * 70)
print("DIRECT TEXT → CAD EVALUATION")
print("=" * 70)
print(f"Query: {query}")


# Get text embedding from Ollama
embedding = get_text_embedding(query)
embedding = np.asarray(embedding, dtype=np.float32)

if len(embedding) != text_dim:
    raise ValueError(
        f"Expected text dimension {text_dim}, got {len(embedding)}"
    )

text_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)


# Project text to CAD latent space
with torch.no_grad():
    query_vector = adapter(text_tensor)

query_vector = (
    query_vector.squeeze(0).cpu().numpy().astype(np.float32)
)
query_vector /= (np.linalg.norm(query_vector) + 1e-8)


# Compute cosine similarity scores
results = []
for model_name, cad_vector in cad_latents.items():
    similarity = float(np.dot(query_vector, cad_vector))
    results.append((model_name, similarity))

results.sort(key=lambda x: x[1], reverse=True)

# Display results
print("\n" + "-" * 70)
print(f"{'Rank':<8}{'CAD Model':<35}{'Similarity':<15}")
print("-" * 70)

for rank, (model, score) in enumerate(results, start=1):
    print(f"{rank:<8}{model:<35}{score * 100:>8.2f}%")

print("-" * 70)
print("\n[SUCCESS] Direct latent evaluation complete.")