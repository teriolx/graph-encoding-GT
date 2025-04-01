import torch
from torch_geometric.graphgym.register import register_edge_encoder, register_node_encoder
from torch_geometric.graphgym import cfg


@register_edge_encoder('DummyEdge')
class DummyEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        batch.edge_attr = self.encoder(dummy_attr)
        return batch
    

@register_node_encoder('DummyNode')
class DummyNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, expand_x=False):
        super().__init__()

        self.enc_dim = cfg.ctenc_DummyNode.dim_ct
        self.emb_dim = emb_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, batch):
        batch.x = torch.cat((batch.x, torch.ones(batch.x.shape[0], self.enc_dim, 
                                                 device=self.device)), 1)
        assert batch.x.shape[1] == self.emb_dim
        return batch
