import networkx as nx
import torch
from torch_geometric.data import Data


class GraphOfWords:
    def __init__(
        self,
        input: torch.Tensor,
        window_size: int = 2,
        embedded: bool = False,
        device="cpu",
    ):
        self.window_size = window_size
        self.input = input
        self.embedded = embedded
        self.device = device

    def _create_graph(self) -> nx.Graph:
        if not self.embedded:
            words = self.input.split()
            nodes = words
        else:
            # Assume self.input is a list or tensor of embeddings
            # Use indices as node identifiers
            words = list(range(len(self.input)))
            nodes = words

        g = nx.Graph()
        for i, node in enumerate(nodes):
            g.add_node(node)
            for j in range(1, self.window_size + 1):
                if i - j >= 0:
                    g.add_edge(node, nodes[i - j])
                if i + j < len(nodes):
                    g.add_edge(node, nodes[i + j])
        return g

    def to_pyg_data(self) -> Data:
        g = self._create_graph()
        # Map node labels to integer indices
        node2idx = {node: idx for idx, node in enumerate(g.nodes())}
        # Convert edges to index pairs
        edge_index = (
            torch.tensor(
                [[node2idx[u], node2idx[v]] for u, v in g.edges()], dtype=torch.long
            )
            .t()
            .contiguous()
            .to(self.device)
        )
        # Use embeddings as node features if available
        if self.embedded:
            x = torch.tensor(self.input, dtype=torch.float).to(self.device)
        else:
            x = (
                torch.tensor([1] * g.number_of_nodes(), dtype=torch.float)
                .view(-1, 1)
                .to(self.device)
            )
        data = Data(x=x, edge_index=edge_index)

        return data
