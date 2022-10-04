from .base import BaseModel
from torch import nn as nn
from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as
from .ae import AutoEncoder
import torch

class BERTLSModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        max_len = args.bert_max_len
        num_items = args.num_items
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout
        self.device = args.device

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

        self.out = nn.Linear(hidden, args.num_items + 1)

        self.ae = AutoEncoder(args)

    @classmethod
    def code(cls):
        return 'bert_ls'

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        emb = self.embedding(x)

        # passing through AutoEncoder
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        ae_output, ae_latent = self.ae(emb.reshape(-1, emb.shape[-1]))
        ae_output = ae_output.reshape(batch_size, seq_len, -1)
        ae_latent = ae_latent.reshape(batch_size, seq_len, -1)
        x = emb.reshape(batch_size, seq_len, -1)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return self.out(x), emb, ae_latent, ae_output
