from torch_geometric.data import Data
from torch.nn import Module
import torch
import networkx as nx

class GraphToData(Module):
    def __init__(self, gow_model: Module):
        super().__init__()
        self.gow_model = gow_model

    def _to_pyg_data(self, g: nx.Graph) -> Data:
        node_list = list(g.nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        features = [g.nodes[node]['feature'] for node in node_list]
        x = torch.stack(features)

        edges = []
        for u, v in g.edges:
            edges.append((node_to_idx[u], node_to_idx[v]))
            edges.append((node_to_idx[v], node_to_idx[u]))  # undirected -> to be decided if needed
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def forward(self, text: str) -> Data:
        nx_graph = self.gow_model(text)
        return self._to_pyg_data(nx_graph)
