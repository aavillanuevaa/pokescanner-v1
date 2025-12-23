# embedder.py
import numpy as np
import torch
import open_clip
from PIL import Image

class ClipEmbedder:
    """
    OpenCLIP image embedder.
    Produces L2-normalized embeddings suitable for cosine similarity.
    """
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_pil(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()

def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int = 3):
    """
    query: (D,) normalized
    matrix: (N, D) normalized
    returns: (indices, scores)
    """
    scores = matrix @ query
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]
