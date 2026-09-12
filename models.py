import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return self.classifier(x)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads))
        for _ in range(1, n_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        if n_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False))
        else:
            self.convs = nn.ModuleList([GATConv(in_dim, hidden_dim, heads=heads)])
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, return_attention_weights=False):
        attns = []
        for i, conv in enumerate(self.convs):
            if return_attention_weights:
                x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
                attns.append((ei, alpha))
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is not None:
            x = global_mean_pool(x, batch)
        out = self.classifier(x)
        if return_attention_weights:
            return out, attns
        return out


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return self.classifier(x)


MODELS = {"gcn": GCN, "gat": GAT, "sage": GraphSAGE}
