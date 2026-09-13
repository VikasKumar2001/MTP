import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from text_adapter import TextToCADAdapter
from ollama_client import get_text_embedding

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR_FILE = os.path.join(PROJECT_ROOT, "text_to_cad", "text_cad_pairs.json")
CAD_LATENT_DIR = os.path.join(PROJECT_ROOT, "text_to_cad", "cad_latents")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "weights", "text_to_cad.pth")

# Configuration
CAD_DIM = 512
EPOCHS = 200
LEARNING_RATE = 1e-4
TEMPERATURE = 0.07
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_pairs():
    """Load text-CAD pairs from JSON file."""
    with open(PAIR_FILE, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    if len(pairs) == 0:
        raise RuntimeError("text_cad_pairs.json is empty.")

    return pairs


def load_cad_latents(pairs):
    """Load CAD latent vectors from .npy files."""
    cad_latents = {}
    cad_files = sorted(set(pair["cad"] for pair in pairs))

    for cad_file in cad_files:
        filename = os.path.splitext(cad_file)[0] + ".npy"
        path = os.path.join(CAD_LATENT_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"CAD latent not found: {path}")

        latent = np.load(path)

        if latent.shape != (CAD_DIM,):
            raise ValueError(
                f"Invalid latent shape for {cad_file}: {latent.shape}"
            )

        cad_latents[cad_file] = torch.tensor(latent, dtype=torch.float32)

    return cad_latents


def generate_text_embeddings(pairs):
    """Generate Ollama embeddings for all text queries."""
    embeddings = []

    print("\nGenerating Ollama embeddings...")
    print("-" * 70)

    for i, pair in enumerate(pairs):
        text = pair["text"]
        print(f"[{i + 1:03d}/{len(pairs):03d}] {text}")

        embedding = get_text_embedding(text)
        embedding = torch.tensor(embedding, dtype=torch.float32)
        embeddings.append(embedding)

    embeddings = torch.stack(embeddings)
    return embeddings


# ============================================================
# MULTI-POSITIVE CONTRASTIVE LOSS
# ============================================================

def multi_positive_contrastive_loss(
    text_vectors,
    cad_vectors,
    target_indices,
    temperature=0.07
):
    """
    Multiple text queries can belong to the same CAD model.

    Example:

        "office chair"       ─┐
        "desk chair"          ├──> Office chair.stl
        "work chair"         ─┘

    Therefore, all text queries mapped to the same CAD
    model are treated as positive examples.
    """

    text_vectors = F.normalize(
        text_vectors,
        p=2,
        dim=1
    )

    cad_vectors = F.normalize(
        cad_vectors,
        p=2,
        dim=1
    )

    # --------------------------------------------------------
    # Similarity between every text query and every CAD model
    # --------------------------------------------------------

    logits = (
        text_vectors @ cad_vectors.T
    ) / temperature

    # --------------------------------------------------------
    # For each text query, select its correct CAD model
    # --------------------------------------------------------

    target_indices = target_indices.to(
        logits.device
    )

    loss = F.cross_entropy(
        logits,
        target_indices
    )

    return loss


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("TEXT → CAD ADAPTER TRAINING")
    print("=" * 70)

    # Load pairs
    pairs = load_pairs()
    print(f"[INFO] Training text queries: {len(pairs)}")

    # Load unique CAD latents
    cad_latents = load_cad_latents(pairs)
    cad_files = sorted(cad_latents.keys())
    print(f"[INFO] Unique CAD models: {len(cad_files)}")

    # Generate text embeddings
    text_embeddings = generate_text_embeddings(pairs)
    text_dim = text_embeddings.shape[1]
    print(f"\n[INFO] Ollama embedding dimension: {text_dim}")

    # Build CAD tensor
    cad_tensor = torch.stack([cad_latents[cad_file] for cad_file in cad_files])
    cad_tensor = F.normalize(cad_tensor, p=2, dim=1)

    # Create target index for every text query
    cad_to_index = {cad_file: i for i, cad_file in enumerate(cad_files)}
    target_indices = torch.tensor(
        [cad_to_index[pair["cad"]] for pair in pairs], dtype=torch.long
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Move data to device
    text_embeddings = text_embeddings.to(device)
    cad_tensor = cad_tensor.to(device)
    target_indices = target_indices.to(device)

    # Create adapter
    model = TextToCADAdapter(text_dim=text_dim, cad_dim=CAD_DIM).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    print("\nStarting training...")
    print("-" * 70)

    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # Text → 512-D CAD space
        predicted = model(text_embeddings)

        # Contrastive objective
        loss = multi_positive_contrastive_loss(
            predicted, cad_tensor, target_indices, TEMPERATURE
        )

        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d}/{EPOCHS} | Loss: {loss.item():.6f}")

    # Save adapter
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "text_dim": text_dim,
        "cad_dim": CAD_DIM,
        "temperature": TEMPERATURE,
    }

    torch.save(checkpoint, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"[SUCCESS] Adapter saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()