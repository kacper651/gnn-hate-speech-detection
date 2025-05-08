import networkx as nx
import torch
from torch.nn import Module

class GraphOfWords(Module):
    def __init__(self, embedding_model: Module, window_size: int = 2):
        super().__init__()
        self.embedding_model = embedding_model
        self.window_size = window_size

    def forward(self, text: str) -> nx.Graph:
        words = text.lower().split()
        graph = nx.Graph()
        vocab = list(dict.fromkeys(words))

        for word in vocab:
            if hasattr(self.embedding_model, 'model') and hasattr(self.embedding_model.model, '__contains__'):
                if word in self.embedding_model.model:
                    vector = torch.tensor(self.embedding_model.model[word], dtype=torch.float32)
                else:
                    continue # skip words not in the model
            else:
                vector = self.embedding_model(word)
            graph.add_node(word, feature=vector)

        for i, word in enumerate(words):
            for j in range(i + 1, min(i + self.window_size + 1, len(words))):
                if word != words[j] and word in graph and words[j] in graph:
                    graph.add_edge(word, words[j])

        return graph
