import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
import random

def lock_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

lock_seeds(42)

# ==========================================================
# 1. ARCHITECTURES
# ==========================================================
class PointNetFeatureExtractor(nn.Module):
    """Your exact architecture from pointnet.py"""
    def __init__(self):
        super(PointNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        return x.view(-1, 1024)

class PointNetDecoder(nn.Module):
    """Temporary decoder to force self-supervised learning"""
    def __init__(self, num_points=4096):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_points * 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 3, self.num_points)

class PointNetAutoencoder(nn.Module):
    def __init__(self):
        super(PointNetAutoencoder, self).__init__()
        self.encoder = PointNetFeatureExtractor()
        self.decoder = PointNetDecoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# ==========================================================
# 2. DATASET
# ==========================================================
class RawCADDataset(Dataset):
    def __init__(self, data_dir="../Data", num_points=4096):
        self.files = glob.glob(os.path.join(data_dir, "*.stl"))
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lock_seeds(idx) # Isolate randomness per item
        mesh = trimesh.load(self.files[idx], force="mesh")
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        
        # Normalize
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
            
        # Shape: (3, 4096)
        points_tensor = torch.tensor(points, dtype=torch.float32).transpose(0, 1)
        return points_tensor

# ==========================================================
# 3. CHAMFER LOSS & TRAINING LOOP
# ==========================================================
def chamfer_distance(p1, p2):
    # Inputs: (Batch, 3, NumPoints) -> transpose to (Batch, NumPoints, 3)
    p1 = p1.transpose(1, 2)
    p2 = p2.transpose(1, 2)
    
    diff = p1.unsqueeze(2) - p2.unsqueeze(1)
    dist_sq = torch.sum(diff ** 2, dim=-1)
    
    min_p1_to_p2, _ = torch.min(dist_sq, dim=2)
    min_p2_to_p1, _ = torch.min(dist_sq, dim=1)
    return torch.mean(min_p1_to_p2) + torch.mean(min_p2_to_p1)

def train_pointnet(epochs=50, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Pre-training PointNet on {device}...")

    dataset = RawCADDataset(data_dir="Data") # Assuming run from project root
    if len(dataset) == 0:
        print("[ERROR] No STLs found in Data/. Check path.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PointNetAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_points in dataloader:
            batch_points = batch_points.to(device)
            optimizer.zero_grad()
            
            reconstructed = model(batch_points)
            loss = chamfer_distance(batch_points, reconstructed)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Chamfer Loss: {total_loss/len(dataloader):.6f}")

    os.makedirs("weights", exist_ok=True)
    save_path = "weights/pointnet_weights.pth"
    torch.save(model.encoder.state_dict(), save_path)
    print(f"\n[SUCCESS] Trained PointNet encoder saved to {save_path}")

if __name__ == "__main__":
    train_pointnet()