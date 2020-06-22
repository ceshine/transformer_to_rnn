import warnings
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regularizations import WeightDrop, LockedDropout, EmbeddingDropout
from .attentions import Attention

def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    return h.detach() if type(h) == torch.Tensor else tuple(repackage_var(v) for v in h)


class RNNStack(nn.Module):
    """ A stack of LSTM or QRNN layers to drive the network, and
        variational dropouts in the embedding and LSTM/QRNN layers

        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    def __init__(self, emb_sz: int, n_hid: int, n_layers: int, bidir=False,
                 dropouth=0.3, dropouti=0.65, wdrop=0.5, unit_type="lstm", qrnn=False):
        """Default constructor for the RNNStack class

        Parameters
        ----------
        emb_sz : int
            the embedding size used to encode each token
        n_hid : int
            number of hidden units per layer.
        n_layers : int
            number of layers to use in the architecture.
        bidir : bool, optional
            Use bidirectional layout. (only used when qrnn=False)
        dropouth : float, optional
            dropout to apply to the activations going from one layer to another.
        dropouti : float, optional
            dropout to apply to the input layer.
        wdrop : float, optional
            dropout used for a LSTM's internal (or hidden) recurrent weights.
            (only used when qrnn=False)
        unit_type: str, optional
            RNN unit. Available: "lstm", "gru", "qrnn"(require installing the qrnn package)
        """
        super().__init__()
        unit_type = unit_type.strip().lower()
        self.qrnn = unit_type == "qrnn"
        self.unit = nn.LSTM if unit_type == "lstm" else nn.GRU
        self.bs = 1
        self.ndir = 2 if bidir else 1
        assert not (
            self.qrnn and self.bidir
        ), "QRNN does not support bidirectionality."
        if self.qrnn:
            # Using QRNN requires cupy: https://github.com/cupy/cupy
            from .torchqrnn.qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, n_hid,
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(n_layers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.linear = WeightDrop(
                        rnn.linear, wdrop, layer_names=['weight'])
        else:
            self.rnns = [self.unit(emb_sz if l == 0 else n_hid, n_hid // self.ndir,
                                   1, bidirectional=bidir) for l in range(n_layers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.emb_sz, self.n_hid, self.n_layers = emb_sz, n_hid, n_layers
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList(
            [LockedDropout(dropouth) for l in range(n_layers)])

    def forward(self, emb):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl, bs, emb_sz = emb.size()
        assert emb_sz == self.emb_sz, "input size does not match model size"
        if bs != self.bs:
            self.bs = bs
            self.reset()
        with torch.set_grad_enabled(self.training):
            raw_output = self.dropouti(emb)
            new_hidden, raw_outputs = [], []
            for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.n_layers - 1:
                    raw_output = drop(raw_output)
            self.hidden = repackage_var(new_hidden)
        return raw_outputs, self.hidden

    def one_hidden(self, l):
        nh = self.n_hid // self.ndir
        return next(self.parameters()).new_empty(self.ndir, self.bs, nh).zero_()

    def reset(self):
        if self.qrnn or (self.unit is nn.GRU):
            if self.qrnn:
                [r.reset() for r in self.rnns]
            self.hidden = [self.one_hidden(l) for l in range(self.n_layers)]
        else:
            self.hidden = [(self.one_hidden(l), self.one_hidden(l))
                           for l in range(self.n_layers)]


class LinearBlock(nn.Module):
    """Simple Linear Block with Dropout and BatchNorm

    Adapted from fast.ai v0.7
    """
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)
        nn.init.kaiming_normal_(self.lin.weight)
        nn.init.constant_(self.lin.bias, 0)
        nn.init.ones_(self.bn.weight)

    def forward(self, x):
        return self.lin(self.drop(self.bn(x)))


class PoolingFCN(nn.Module):
    """FCN that make use of all sequence inputs.

    Average pooling + max pooling + last time step.

    Adapted from fast.ai v0.7
    """
    def __init__(self, layers: List[int], drops: Sequence[float], bidir: bool = False):
        super().__init__()
        assert len(layers) == len(drops) + 1
        self.bidir = bidir
        layers[0] = layers[0] * 3
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(bs, -1)

    def forward(self, input_tensors: torch.FloatTensor, input_lengths: torch.LongTensor):
        output = input_tensors[-1]
        _, bs, _ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        if self.bidir:
            n_hid = output.size(2) // 2
            tails = []
            for i in range(len(input_lengths)):
                tails.append(output[input_lengths[i]-1, i, :n_hid])
            x = torch.cat(
                [torch.stack(tails), output[0, :, n_hid:],
                 mxpool, avgpool], 1)
        else:
            x = torch.cat([output[-1], mxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x


class AttentionFCN(nn.Module):
    """FCN that make use of all sequence inputs.
    """
    def __init__(self, layers: List[int], drops: Sequence[float]):
        super().__init__()
        assert len(layers) == len(drops) + 1
        self.attention = Attention(layers[0], batch_first=False)
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def forward(self, input_tensors: torch.FloatTensor, input_lengths: torch.LongTensor):
        output = input_tensors[-1]
        output = self.attention(output, input_lengths)[0]
        for l in self.layers:
            l_x = l(output)
            output = F.relu(l_x)
        return l_x


class BasicEmbeddings(nn.Module):
    """A simple wrapper around an embeddings matrix
       that comes with optional embedding dropouts
    """
    initrange = 0.1

    def __init__(self, voc_sz: int, emb_sz: int, pad_idx: int, dropoute: float = 0):
        """Default constructor for the BasicEmbeddings class

        Parameters
        ----------
        voc_sz : int
            number of vocabulary (or tokens) in the source dataset.
        emb_sz : int
            the embedding size used to encode each token.
        pad_idx : int
            the int value used for padding text.
        dropoute : float, optional
            dropout to apply to the embedding layer. (the default is 0)
        """
        super().__init__()
        self.voc_sz, self.emb_sz, self.dropoute = voc_sz, emb_sz, dropoute
        self.encoder: nn.Module = nn.Embedding(voc_sz, emb_sz, padding_idx=pad_idx)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)        
        if dropoute > 0:
            self.encoder = EmbeddingDropout(self.encoder, dropoute)

    def forward(self, input_tensor: torch.LongTensor):
        """ Invoked during the forward propagation of the BasicEmbeddings module.

        Parameters
        ----------
        input_tensor: torch.Tensor
            A Long Tensor with shape (seq_length, batch_size)
        """
        return self.encoder(input_tensor)