import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class FusionMLP(nn.Module):
        """
        A standard 3-layer MLP to map concatenated multimodal features 
        down to a target latent dimension.
        """
        def __init__(self, input_dim: int, target_dim: int):
            super(FusionMLP, self).__init__()
            hidden_dim = (input_dim + target_dim) // 2
            
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, target_dim),
                # LayerNorm helps stabilize the final latent vector distribution 
                # which is beneficial for the QR encoder's consistency
                nn.LayerNorm(target_dim) 
            )

        def forward(self, x):
            return self.network(x)


class FeatureFusion:
    """
    Milestone 9: Fuses all independent descriptors into a single latent representation
    using an MLP dimensionality reduction network.
    """
    def __init__(self, target_dim: int = 512):
        self.target_dim = target_dim
        self.mlp = None
        
        if HAS_TORCH:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None

    def fuse(self, extracted_features: dict) -> list:
        fused_vector = []
        
        # M4: Geometry
        if "geometry_vector" in extracted_features:
            fused_vector.extend(extracted_features["geometry_vector"])
            
        # M2: Slice Structural Signatures
        if "slice_vector" in extracted_features:
            fused_vector.extend(extracted_features["slice_vector"])
            
        # M5: ShapeDNA
        if "shape_dna_vector" in extracted_features:
            fused_vector.extend(extracted_features["shape_dna_vector"])
            
        # M6: FPFH
        if "fpfh_vector" in extracted_features:
            fused_vector.extend(extracted_features["fpfh_vector"])
            
        # M7: PointNet
        if "pointnet_vector" in extracted_features:
            fused_vector.extend(extracted_features["pointnet_vector"])
            
        # M8: DINOv2
        if "dinov2_vector" in extracted_features:
            fused_vector.extend(extracted_features["dinov2_vector"])
        
        return self._apply_mlp(fused_vector)

    def _apply_mlp(self, vector: list) -> list:
        """
        Passes the concatenated vector through the PyTorch MLP.
        Automatically loads trained weights if they exist.
        """
        if not HAS_TORCH:
            print("[WARNING] PyTorch not found. Falling back to simple truncation/padding.")
            return self._fallback_projection(vector)

        input_dim = len(vector)
        if input_dim == 0:
            return [0.0] * self.target_dim

        # Lazy initialization: Build the MLP based on the concatenated vector size
        if self.mlp is None or self.mlp.network[0].in_features != input_dim:
            self.mlp = FusionMLP(input_dim=input_dim, target_dim=self.target_dim).to(self.device)
            
            # --- INTEGRATED WEIGHT LOADING FOR ENCODER ---
            weights_path = "weights/encoder.pth"
            if os.path.exists(weights_path):
                try:
                    self.mlp.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                    print(f"[FeatureFusion] Successfully loaded trained weights from {weights_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load encoder weights: {e}. Output will be random.")
            else:
                print(f"[WARNING] Encoder weights not found at {weights_path}. Output will be untrained.")
                
            self.mlp.eval()  # Set to inference mode

        try:
            # Convert to tensor and add batch dimension: shape (1, input_dim)
            input_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_tensor = self.mlp(input_tensor)
            
            result = output_tensor.squeeze(0).cpu().tolist()
            return [round(v, 6) for v in result]
            
        except Exception as e:
            print(f"[ERROR] MLP Projection failed: {e}")
            return self._fallback_projection(vector)

    def _fallback_projection(self, vector: list) -> list:
        """Original padding/truncation logic used in M3-M8 as a safe fallback."""
        projected = vector[:self.target_dim]
        while len(projected) < self.target_dim:
            projected.append(0.0)
        return projected