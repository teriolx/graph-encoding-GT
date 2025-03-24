import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder)


@register_node_encoder('LPCAEnc')
class LPCAEncoder(torch.nn.Module):
    def __init__(self, emb_dim, expand_x=False):
        super().__init__()


    def forward(self, batch):
        batch.x = torch.cat((batch.x, getattr(batch, 'counts')), 1)
        return batch
