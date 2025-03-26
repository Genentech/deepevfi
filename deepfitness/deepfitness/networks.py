from loguru import logger
import numpy as np

import torch

from hackerargs import args
from deepfitness.data.databatches import DataBatch
from deepfitness.genotype import featurizers


class Transformer(torch.nn.Module):
    def __init__(self):
        """ Transformer encoder: variable-len seq -> scalar """
        super().__init__()
        alphabet_size = len(featurizers.alphabets[args.get('ft.alphabet')])
        alphabet_size_w_padding = alphabet_size + 1

        embed_dim = args.setdefault('net.transformer_embed_dim', 16)
        self.embedder = torch.nn.Embedding(
            num_embeddings = alphabet_size_w_padding,
            embedding_dim = embed_dim,
            padding_idx = -1,
        )
        self.pos_encoder = PositionalEncoding(
            d_model = embed_dim,
        )
        self.transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = args.setdefault('net.transformer_nhead', 8),
            batch_first = True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer = self.transformer_enc_layer,
            num_layers = args.setdefault('net.transformer_num_layers', 4),
        )
        self.linear_layer = torch.nn.Linear(
            in_features = embed_dim,
            out_features = 1,
        )

    def forward(self, batch: DataBatch) -> torch.Tensor:
        """ batch
            -----
            genotype_tensor: B x L, int idxs
            count: B x 1, int 
            next_count: B x 1, int
            padding_mask: B x L
        """
        genotype_tensor = batch['genotype_tensor']
        padding_mask = batch['padding_mask']

        # input: B x L
        x = self.embedder(genotype_tensor)
        # B x L x E

        x = self.pos_encoder(x)
        # B x L x E

        x = self.transformer_encoder.forward(
            src = x,
            src_key_padding_mask = padding_mask,
        )
        # B x L x E

        # aggregate
        x = torch.mean(x, dim = 1)
        # B x E

        x = self.linear_layer(x)
        # B x 1
        return x
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: Tensor, shape ``[batch_size, seq_len, embedding_dim]`` """
        return x + self.pe[:, :x.size(1)]


class MLP(torch.nn.Module):
    def __init__(self):
        """ Multilayer perceptron """
        super().__init__()

        inp_dim_arg = 'net.mlp_inp_dim'
        if inp_dim_arg not in args:
            logger.error(f'Using MLP requires {inp_dim_arg} as arg')
        inp_dim = args.get(inp_dim_arg)
        
        hidden_dim = args.setdefault('net.mlp_hidden_dim', 64)
        n_layers = args.setdefault('net.mlp_n_layers', 3)
        with_bn = args.setdefault('net.mlp_batch_norm', True)
        p_dropout = args.setdefault('net.mlp_dropout', 0.1)
        out_dim = 1

        self.mlp = torch.nn.ModuleList()
        i_h_dims = [inp_dim] + [hidden_dim for _ in range(n_layers)]
        for i, o in zip(i_h_dims[:-1], i_h_dims[1:]):
            self.mlp.append(torch.nn.Linear(i, o))

            if with_bn:
                self.mlp.append(torch.nn.BatchNorm1d(o))

            self.mlp.append(torch.nn.LeakyReLU(inplace = True))

            if p_dropout > 0:
                self.mlp.append(torch.nn.Dropout(p = p_dropout))

        self.mlp.append(torch.nn.Linear(i_h_dims[-1], out_dim))

    def forward(self, batch: DataBatch) -> torch.Tensor:
        out = batch['genotype_tensor']
        for layer in self.mlp:
            out = layer(out)
        return out


# getter
def get_network(name: str) -> torch.nn.Module:
    name_to_network = {
        'transformer': Transformer,
        'mlp': MLP,
    }
    return name_to_network[name]