import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DINOv2Descriptor:
    """
    DINOv2 ViT-B/14 visual descriptor.

    Extracts a 768-D embedding from each rendered CAD view
    and averages the embeddings across all views.
    """

    def __init__(self, target_dim: int = 768):

        self.target_dim = target_dim
        self.model = None
        self.processor = None

        if not HAS_TORCH:
            print("[WARNING] PyTorch/Transformers not available.")
            return

        # Select GPU if available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[DINOv2] Device: {self.device}")

        try:

            model_name = "facebook/dinov2-base"

            print("[DINOv2] Loading processor...")

            self.processor = AutoImageProcessor.from_pretrained(
                model_name
            )

            print("[DINOv2] Loading model...")

            self.model = AutoModel.from_pretrained(
                model_name
            )

            self.model = self.model.to(self.device)
            self.model.eval()

            print("[DINOv2] Model loaded successfully.")
            print("[DINOv2] Embedding dimension: 768")

        except Exception as e:

            self.model = None
            self.processor = None

            print(
                f"[WARNING] DINOv2 model failed to load: {e}"
            )

    def extract(self, images: list) -> dict:

        # Safety check
        if (
            not HAS_TORCH
            or self.model is None
            or self.processor is None
            or not images
        ):
            return {
                "dinov2_vector": [0.0] * self.target_dim
            }

        try:

            embeddings = []

            with torch.no_grad():

                for img in images:

                    # Handle PIL image
                    if isinstance(img, Image.Image):
                        image = img.convert("RGB")

                    # Handle image path
                    else:
                        image = Image.open(img).convert("RGB")

                    # DINOv2 preprocessing
                    inputs = self.processor(
                        images=image,
                        return_tensors="pt"
                    )

                    # Move tensors to GPU/CPU
                    inputs = {
                        key: value.to(self.device)
                        for key, value in inputs.items()
                    }

                    # Forward pass
                    outputs = self.model(**inputs)

                    # CLS token
                    feature = outputs.last_hidden_state[:, 0, :]

                    # Convert to numpy
                    feature = (
                        feature
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

                    embeddings.append(feature)

            # Average all rendered views
            global_embedding = np.mean(
                embeddings,
                axis=0
            )

            # Make sure output is exactly 768-D
            if len(global_embedding) != self.target_dim:

                print(
                    f"[WARNING] Expected {self.target_dim} dimensions "
                    f"but got {len(global_embedding)}"
                )

                return {
                    "dinov2_vector": [0.0] * self.target_dim
                }

            result = global_embedding.tolist()

            return {
                "dinov2_vector": [
                    round(float(v), 6)
                    for v in result
                ]
            }

        except Exception as e:

            print(
                f"[ERROR] DINOv2 extraction failed: {e}"
            )

            return {
                "dinov2_vector": [0.0] * self.target_dim
            }