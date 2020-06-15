from typing import Sequence

import torch
import torch.nn as nn

from .modules import BasicEmbeddings, RNNStack, AttentionFCN, PoolingFCN

class SequenceModel(nn.Module):
    def __init__(self, embeddings: BasicEmbeddings, encoder: RNNStack, fcn: nn.Module):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.fcn = fcn

    def forward(self, input_tokens: torch.LongTensor, input_lengths: torch.LongTensor):
        # input_tokens shape (seq_length, batch_size)
        embeddings = self.embeddings(input_tokens)
        # embeddings shape (seq_length, batch_size, emb_sz)
        # Remember to reset the hidden states!
        self.encoder.reset()
        rnn_output, _ = self.encoder(embeddings)
        # rnn_output shape (seq_length, batch_size, n_hid)
        outputs = self.fcn(rnn_output, input_lengths)
        # outputs shape (seq_length, batch_size, voc_size)
        return outputs

    def get_layer_groups(self):
        return [self.embeddings, *self.encoder.rnns, self.fcn]


def get_sequence_model(
        voc_size: int,
        emb_size: int,
        pad_idx: int,
        dropoute: float,
        rnn_hid: int,
        rnn_layers: int,
        bidir: bool,
        dropouth: float,
        dropouti: float,
        wdrop: float,
        unit_type: str,
        fcn_layers: Sequence[int],
        fcn_dropouts: Sequence[float],
        use_attention: bool = True):
    embeddings = BasicEmbeddings(voc_size, emb_size, pad_idx, dropoute)
    rnn_stack = RNNStack(emb_size, rnn_hid, rnn_layers, bidir,
                         dropouth, dropouti, wdrop, unit_type=unit_type)
    if use_attention:
        fcn: nn.Module = AttentionFCN([rnn_hid] + list(fcn_layers), fcn_dropouts)
    else:
        fcn = PoolingFCN([rnn_hid] + list(fcn_layers), fcn_dropouts, bidir)
    return SequenceModel(embeddings, rnn_stack, fcn)
    