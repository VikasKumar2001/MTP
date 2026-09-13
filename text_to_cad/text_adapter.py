import torch
import torch.nn as nn
import torch.nn.functional as F


class TextToCADAdapter(nn.Module):
    """Maps an Ollama text embedding into the existing 512-dimensional CAD latent space."""

    def __init__(self, text_dim, cad_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, cad_dim),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: [batch_size, text_dim]

        Returns
        -------
        torch.Tensor
            Shape: [batch_size, 512]
        """
        x = self.network(x)
        # Normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        return x