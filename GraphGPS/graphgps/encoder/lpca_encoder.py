import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder)
import numpy as np


@register_node_encoder('LPCAEnc')
class LPCAEncoder(torch.nn.Module):
    def __init__(self, emb_dim, expand_x=False):
        super().__init__()

        self.enc_dim = cfg.ctenc_LPCAEnc.dim_ct
        self.emb_dim = emb_dim


    def forward(self, batch):
        batch.x = torch.cat((batch.x, getattr(batch, 'lpca_enc')), 1)
        assert batch.x.shape[1] == self.emb_dim
        return batch
