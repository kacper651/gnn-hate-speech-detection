import torch
import numpy as np
from gensim.models import KeyedVectors
from torch.nn import Module

class Word2VecEmbedding(Module):
    def __init__(self, w2v_path, device='cpu'):
        super().__init__()
        self.device = device
        self.model = KeyedVectors.load(w2v_path)
        self.embedding_dim = self.model.vector_size

    def forward(self, text: str) -> torch.Tensor:
        words = text.lower().split()
        valid_vectors = [
            self.model[word] for word in words if word in self.model
        ]
        if not valid_vectors:
            return torch.zeros(self.embedding_dim).to(self.device)

        avg_vector = np.mean(valid_vectors, axis=0)
        return torch.tensor(avg_vector, dtype=torch.float32).to(self.device)
