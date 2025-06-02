import torch
import numpy as np
from gensim.models import Word2Vec

class Word2VecEmbedding():
    def __init__(self, w2v_path, device='cpu'):
        self.device = device
        self.model = Word2Vec.load(w2v_path)

    def _get_embedding(self, word):
        if word in self.model.wv.index_to_key:
            return torch.tensor(self.model.wv[word], dtype=torch.float32, device=self.device)
        else:
            return None
        
    def _get_embeddings(self, words):
        embeddings = []
        for word in words.split():
            embedding = self._get_embedding(word)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                print(f"[SKIP] Word '{word}' not found in the model")   
        return torch.stack(embeddings) if embeddings else None