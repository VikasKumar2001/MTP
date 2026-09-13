import os
import sys
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ==========================================================
# DETERMINISTIC SEED LOCK
# ==========================================================
def lock_seeds(seed=42):
    """Forces strict determinism for reproducible latent spaces."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Lock globally immediately upon script execution
lock_seeds(42)

# ==========================================================
# PROJECT ROOT
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.autoencoder import CADAutoencoder

# ==========================================================
# CONFIGURATION
# ==========================================================
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset.pt")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
ENCODER_PATH = os.path.join(WEIGHTS_DIR, "encoder.pth")
DECODER_PATH = os.path.join(WEIGHTS_DIR, "decoder.pth")

# ==========================================================
# TRAINER
# ==========================================================
class CADTrainer:
    def __init__(
        self,
        input_dim=1925,
        latent_dim=512,
        num_vertices=642,
        epochs=200,
        batch_size=8,
        learning_rate=1e-4
    ):
        # Re-lock before model initialization
        lock_seeds(42)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_vertices = num_vertices
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TRAINING] Device: {self.device}")

        if self.device.type == "cuda":
            print(f"[TRAINING] GPU: {torch.cuda.get_device_name(0)}")
            
        print(f"[INFO] Target vertices: {self.num_vertices}")

        # --------------------------------------------------
        # Model
        # --------------------------------------------------
        self.model = CADAutoencoder(
            input_feature_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_vertices=self.num_vertices
        ).to(self.device)

        # --------------------------------------------------
        # Optimizer & Loss
        # --------------------------------------------------
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.criterion = nn.MSELoss()

    # ======================================================
    # LOAD DATASET
    # ======================================================
    def load_dataset(self):
        print(f"[TRAINING] Loading dataset from:\n           {DATASET_PATH}")

        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(
                f"Dataset not found:\n{DATASET_PATH}\n\n"
                f"Run prepare_dataset.py first."
            )

        data = torch.load(DATASET_PATH, weights_only=True)

        inputs = data["inputs"]
        targets = data["targets"]

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)

        inputs = inputs.float()
        targets = targets.float()

        dataset = TensorDataset(inputs, targets)
        print(f"[TRAINING] Dataset samples: {len(dataset)}")
        
        return dataset

    # ======================================================
    # TRAIN
    # ======================================================
    def train(self):
        dataset = self.load_dataset()

        # Re-lock before DataLoader to ensure identical batch shuffling
        lock_seeds(42)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        print("\n" + "=" * 60)
        print(f"[TRAINING] Starting on {self.device} for {self.epochs} epochs...")
        print("=" * 60)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            for descriptors, gt_points in dataloader:
                descriptors = descriptors.to(self.device)
                gt_points = gt_points.to(self.device)

                latent, predicted_points = self.model(descriptors)

                loss = self.criterion(predicted_points, gt_points)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            average_loss = (epoch_loss / batch_count) if batch_count > 0 else 0.0

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:03d}/{self.epochs:03d} | Loss: {average_loss:.8f}")

        # ======================================================
        # TRAINING COMPLETE
        # ======================================================
        print("\n" + "=" * 60)
        print("[TRAINING] Training completed deterministically.")

        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        torch.save(self.model.encoder.state_dict(), ENCODER_PATH)
        torch.save(self.model.decoder.state_dict(), DECODER_PATH)

        print(f"[SUCCESS] Fixed Encoder saved to:\n           {ENCODER_PATH}")
        print(f"[SUCCESS] Fixed Decoder saved to:\n           {DECODER_PATH}")
        print("=" * 60)

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    trainer = CADTrainer(
        input_dim=1925,
        latent_dim=512,
        num_vertices=642,
        epochs=200,
        batch_size=8,
        learning_rate=1e-4
    )
    trainer.train()