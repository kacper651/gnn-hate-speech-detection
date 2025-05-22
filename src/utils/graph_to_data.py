from typing import Optional
from torch_geometric.data import Data
from torch.nn import Module
import torch
import networkx as nx

class GraphToData(Module):
    def __init__(self, gow_model: Module):
        super().__init__()
        self.gow_model = gow_model

    def _to_pyg_data(self, g: nx.Graph) -> Optional[Data]:
        if g.number_of_nodes() == 0:
            return None  # skip empty graphs

        node_list = list(g.nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        features = []
        for node in node_list:
            feature = g.nodes[node].get('feature')
            if feature is not None:
                features.append(feature)
            else:
                # If any feature is missing, skip the node entirely
                continue

        if not features:
            return None  # avoid crash on torch.stack([])

        x = torch.stack(features)

        # Edges: skip if graph has no valid edges
        edges = []
        for u, v in g.edges:
            if u in node_to_idx and v in node_to_idx:
                edges.append((node_to_idx[u], node_to_idx[v]))
                edges.append((node_to_idx[v], node_to_idx[u]))  # For undirected

        if not edges:
            # Add self-loops to keep the graph valid
            edges = [(i, i) for i in range(len(features))]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)


    def forward(self, text: str) -> Optional[Data]:
        nx_graph = self.gow_model(text)
        return self._to_pyg_data(nx_graph)

