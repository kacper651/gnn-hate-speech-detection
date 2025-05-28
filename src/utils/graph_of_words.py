import networkx as nx
import torch
from torch.nn import Module

class GraphOfWords():
    def __init__(self, input, window_size: int = 2, embedded: bool = False, device = 'cpu'):
        self.window_size = window_size
        self.input = input
        self.embedded = embedded
        self.device = device

    def _create_graph(self) -> nx.Graph:
        words = self.input.split()
        g = nx.Graph()
        
        for i, word in enumerate(words):
            g.add_node(word)
            for j in range(1, self.window_size + 1):
                if i - j >= 0:
                    g.add_edge(word, words[i - j])
                if i + j < len(words):
                    g.add_edge(word, words[i + j])
        return g
    
    def to_pyg_data(self) -> torch.Tensor:
        g = self._create_graph()
        adj_matrix = nx.to_numpy_array(g)
        return torch.tensor(adj_matrix, dtype=torch.float32, device=self.device)
    