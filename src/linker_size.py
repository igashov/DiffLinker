import numpy as np
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from src.egnn import GCL


class DistributionNodes:
    def __init__(self, histogram):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()
        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class SizeGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers, normalization, device='cpu'):
        super(SizeGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.out_node_nf = out_node_nf
        self.device = device

        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.gcl1 = GCL(
            input_nf=self.hidden_nf,
            output_nf=self.hidden_nf,
            hidden_nf=self.hidden_nf,
            normalization_factor=1,
            aggregation_method='sum',
            edges_in_d=1,
            activation=nn.ReLU(),
            attention=False,
            normalization=normalization
        )

        layers = []
        for i in range(n_layers - 1):
            layer = GCL(
                input_nf=self.hidden_nf,
                output_nf=self.hidden_nf,
                hidden_nf=self.hidden_nf,
                normalization_factor=1,
                aggregation_method='sum',
                edges_in_d=1,
                activation=nn.ReLU(),
                attention=False,
                normalization=normalization
            )
            layers.append(layer)

        self.gcl_layers = nn.ModuleList(layers)
        self.embedding_out = nn.Linear(self.hidden_nf, self.out_node_nf)
        self.to(self.device)

    def forward(self, h, edges, distances, node_mask, edge_mask):
        h = self.embedding_in(h)
        h, _ = self.gcl1(h, edges, edge_attr=distances, node_mask=node_mask, edge_mask=edge_mask)
        for gcl in self.gcl_layers:
            h, _ = gcl(h, edges, edge_attr=distances, node_mask=node_mask, edge_mask=edge_mask)

        h = self.embedding_out(h)
        return h
