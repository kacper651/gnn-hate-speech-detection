import networkx as nx
import torch
from torch_geometric.data import Data

class GraphOfWords():
    def __init__(self, input: torch.Tensor, window_size: int = 2, embedded: bool = False, device = 'cpu'):
        self.window_size = window_size
        self.input = input
        self.embedded = embedded
        self.device = device

    def _create_graph(self) -> nx.Graph:
        words = self.input.split() if not self.embedded else self.input
        g = nx.Graph()
        
        for i, word in enumerate(words):
            g.add_node(word)
            for j in range(1, self.window_size + 1):
                if i - j >= 0:
                    g.add_edge(word, words[i - j])
                if i + j < len(words):
                    g.add_edge(word, words[i + j])

        return g
    
    def to_pyg_data(self) -> Data:
        g = self._create_graph()
        edge_index = torch.tensor(list(g.edges), dtype=torch.long).t().contiguous().to(self.device)
        x = torch.tensor([1] * g.number_of_nodes(), dtype=torch.float).view(-1, 1).to(self.device)
        data = Data(x=x, edge_index=edge_index)
        
        return data
    