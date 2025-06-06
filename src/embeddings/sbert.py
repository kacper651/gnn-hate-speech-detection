from torch import Tensor
from sentence_transformers import SentenceTransformer

class SBERTEmbedding():
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_words(self, words: str) -> Tensor:
        words = words.split()
        embeddings = self.model.encode(words, convert_to_tensor=True)
        return embeddings
