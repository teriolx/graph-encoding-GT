import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder)


@register_node_encoder('LPCAEnc')
class LPCAEncoder(torch.nn.Module):
    def __init__(self, emb_dim, expand_x=False):
        super().__init__()

        self.enc_dim = cfg.ctenc_LPCAEnc.dim_ct
        self.emb_dim = emb_dim
        self.pass_as_var = cfg.ctenc_LPCAEnc.pass_as_var if hasattr(cfg.ctenc_LPCAEnc, "pass_as_var") else False


    def forward(self, batch):
        lpca_enc = getattr(batch, 'lpca_enc')

        if self.enc_dim > 0:
            batch.x = torch.cat((batch.x, lpca_enc), 1)
            assert batch.x.shape[1] == self.emb_dim

        if self.pass_as_var:
            # calculate the adjacency matrix scores
            lpca_adj = []
            k = lpca_enc.shape[1] // 2
            for i in range(batch.batch.max().item() + 1):
                node_mask = batch.batch == i
                enc = lpca_enc[node_mask] 

                L, R = enc[:, :k], enc[:, k:]
                adj_i = L @ R.T

                assert adj_i.shape[0] == batch.x[node_mask].shape[0]

                lpca_adj.append(adj_i)
            setattr(batch, 'lpca_adj', torch.block_diag(*lpca_adj))
        return batch
