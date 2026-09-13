import os
import numpy as np
import random
import trimesh

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ==========================================================
# DETERMINISTIC SEED LOCK
# ==========================================================
def lock_seeds(seed=42):
    """Forces PyTorch, NumPy, and Python to be 100% deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

if HAS_TORCH:
    class PointNetFeatureExtractor(nn.Module):
        """
        A streamlined PointNet architecture that extracts a 1024-D global feature 
        vector from a 3D point cloud using shared MLPs (1D Convolutions).
        """
        def __init__(self):
            super(PointNetFeatureExtractor, self).__init__()
            self.conv1 = nn.Conv1d(3, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 1024, 1)
            
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)

        def forward(self, x):
            # Input shape: (Batch_Size, Channels, Num_Points) -> (1, 3, 4096)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            
            # Max pooling across all points to achieve permutation invariance
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            return x

class PointNetDescriptor:
    """
    Extracts a 1024-D PointNet embedding from the CAD model.
    Loads self-supervised pre-trained weights if available.
    """
    def __init__(self, num_points: int = 4096, target_dim: int = 1024):
        self.num_points = num_points
        self.target_dim = target_dim
        
        if HAS_TORCH:
            # Lock seeds right before initializing to ensure deterministic fallback weights
            lock_seeds(42)
            self.model = PointNetFeatureExtractor()
            
            # Determine path to weights (assumes execution from project root)
            weights_path = "weights/pointnet_weights.pth"
            
            if os.path.exists(weights_path):
                print(f"[INFO] PointNetDescriptor: Loading trained weights from {weights_path}")
                self.model.load_state_dict(
                    torch.load(weights_path, weights_only=True)
                )
            else:
                print(f"[WARNING] PointNetDescriptor: {weights_path} not found.")
                print("[WARNING] Using frozen deterministic random weights. Run pretrain_pointnet.py for intelligent extraction.")
            
            self.model.eval()
            
            # Freeze the intelligence to prevent accidental gradient updates
            for param in self.model.parameters():
                param.requires_grad = False

    def extract(self, model_data: dict) -> dict:
        if not HAS_TORCH:
            print("[WARNING] PyTorch not found. Returning zero vector. Install with: pip install torch")
            return {"pointnet_vector": [0.0] * self.target_dim}

        tm = model_data.get("trimesh")
        
        if tm is None or len(tm.faces) == 0:
            print("[WARNING] Invalid mesh for PointNet sampling. Returning zero vector.")
            return {"pointnet_vector": [0.0] * self.target_dim}

        try:
            # Lock seeds right before random surface sampling for perfect reproducibility
            lock_seeds(42)
            
            # 1. Uniformly sample points from the surface of the mesh
            points, _ = trimesh.sample.sample_surface(tm, self.num_points)
            
            # 2. Center and normalize the point cloud into a unit sphere
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
            if max_distance > 0:
                points = points / max_distance
                
            # 3. Convert to PyTorch tensor and reshape for Conv1D
            # Shape becomes: (Batch=1, Channels=3, Points=4096)
            point_cloud_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            
            # 4. Forward pass through PointNet
            with torch.no_grad():
                embedding = self.model(point_cloud_tensor)
                
            result = embedding.squeeze(0).tolist()
            
        except Exception as e:
            print(f"[ERROR] PointNet computation failed: {e}")
            result = [0.0] * self.target_dim

        return {
            "pointnet_vector": [round(v, 6) for v in result]
        }