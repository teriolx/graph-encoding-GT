import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder)


@register_node_encoder('LPCAEnc')
class LPCAEncoder(torch.nn.Module):
    def __init__(self, emb_dim, expand_x=False):
        super().__init__()

        self.enc_dim = cfg.ctenc_LPCAEnc.dim_ct
        self.emb_dim = emb_dim
        self.pass_as_var = cfg.ctenc_LPCAEnc.pass_as_var if hasattr(cfg, "ctenc_LPCAEnc.pass_as_var") else False


    def forward(self, batch):
        lpca_enc = getattr(batch, 'lpca_enc')
        batch.x = torch.cat((batch.x, lpca_enc), 1)
        assert batch.x.shape[1] == self.emb_dim

        if self.pass_as_var:
            setattr(batch, 'lpca_enc', lpca_enc)
        return batch
