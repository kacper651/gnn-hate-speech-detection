from torch.nn import Module, ModuleList, Linear, Dropout, ReLU
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F


class DeepGATClassifier(Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        heads=4,
        dropout=0.5,
        use_global_pool=True,
    ):
        super(DeepGATClassifier, self).__init__()
        self.use_global_pool = use_global_pool
        self.dropout = dropout

        self.convs = ModuleList()

        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )

        self.convs.append(
            GATConv(
                hidden_channels * heads,
                hidden_channels,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        if self.use_global_pool:
            x = global_mean_pool(x, batch)
        else:
            x = x[batch]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x
