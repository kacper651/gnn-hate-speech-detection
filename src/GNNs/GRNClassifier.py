from torch.nn import Linear, Module
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GatedGraphConv
# TODO: fix input size of GatedGraphConv
class GRNClassifier(Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=3):
        super().__init__()
        self.ggnn = GatedGraphConv(out_channels=hidden_channels, num_layers=num_layers)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.ggnn(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)
