from torch import Tensor
from torch.nn import Module
from sentence_transformers import SentenceTransformer

class SBERTEmbedding(Module):
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(model_name).to(device)

    def forward(self, text: str) -> Tensor:
        embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.to(self.device)
