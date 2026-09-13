import torch
import torch.nn as nn
from descriptors.fusion import FusionMLP
from reconstruction.latent_decoder import LatentMeshDecoder

class CADAutoencoder(nn.Module):
    """
    Combines the Fusion Encoder and the Mesh Decoder into a single 
    end-to-end trainable network.
    """
    def __init__(self, input_feature_dim: int, latent_dim: int = 512, num_vertices: int = 642):
        super(CADAutoencoder, self).__init__()
        
        # The Encoder: Maps concatenated descriptors to 512-D
        self.encoder = FusionMLP(input_dim=input_feature_dim, target_dim=latent_dim)
        
        # The Decoder: Maps 512-D to 3D vertex offsets
        self.decoder = LatentMeshDecoder(latent_dim=latent_dim, num_vertices=num_vertices)

    def forward(self, x):
        # x is the raw concatenated descriptor vector
        latent = self.encoder(x)
        offsets = self.decoder(latent)
        return latent, offsets