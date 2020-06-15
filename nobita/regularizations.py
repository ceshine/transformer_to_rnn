import warnings
from typing import Collection, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ArgStar = Collection[Any]


def dropout_mask(x, sz, dropout):
    """ Applies a dropout mask whose size is determined by passed argument 'sz'.
    Args:
        x (torch.Tensor): A torch Variable object
        sz (tuple(int, int, int)): The expected size of the new tensor
        dropout (float): The dropout fraction to apply

    This method uses the bernoulli distribution to decide which activations to keep.
    Additionally, the sampled activations is rescaled is using the factor 1/(1 - dropout).

    In the example given below, one can see that approximately .8 fraction of the
    returned tensors are zero. Rescaling with the factor 1/(1 - 0.8) returns a tensor
    with 5's in the unit places.

    The official link to the pytorch bernoulli function is here:
        http://pytorch.org/docs/master/torch.html#torch.bernoulli

    Examples:
        >>> a_Var = torch.autograd.Variable(torch.Tensor(2, 3, 4).uniform_(0, 1), requires_grad=False)
        >>> a_Var
            Variable containing:
            (0 ,.,.) =
              0.6890  0.5412  0.4303  0.8918
              0.3871  0.7944  0.0791  0.5979
              0.4575  0.7036  0.6186  0.7217
            (1 ,.,.) =
              0.8354  0.1690  0.1734  0.8099
              0.6002  0.2602  0.7907  0.4446
              0.5877  0.7464  0.4257  0.3386
            [torch.FloatTensor of size 2x3x4]
        >>> a_mask = dropout_mask(a_Var.data, (1,a_Var.size(1),a_Var.size(2)), dropout=0.8)
        >>> a_mask
            (0 ,.,.) =
              0  5  0  0
              0  0  0  5
              5  0  5  0
            [torch.FloatTensor of size 1x3x4]
    """
    return x.new_empty(*sz).bernoulli_(1-dropout)/(1-dropout)


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """Locked Dropout / Variational Dropout

        Drops out the same part of the hidden states at each time step.

        Parameters
        ----------
        x : a Tensor of shape (seq_len, batch, input_size)
        """
        if not self.training or not self.p:
            return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return m * x


class WeightDrop(nn.Module):
    """A module that warps another layer in which some weights will be replaced by 0 during training.

    Credit: https://github.com/fastai/fastai/blob/54a9e3cf4fd0fa11fc2453a5389cc9263f6f0d77/fastai/text/models/awd_lstm.py#L27
    """

    def __init__(self, module: nn.Module, weight_p: float, layer_names: Collection[str] = ['weight_hh_l0']):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        self.idxs = [] if hasattr(self.module, '_flat_weights_names') else None
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(
                w, p=self.weight_p, training=False)
            if self.idxs is not None:
                self.idxs.append(self.module._flat_weights_names.index(layer))
        if isinstance(self.module, (nn.RNNBase, nn.modules.rnn.RNNBase)):
            self.module.flatten_parameters = self._do_nothing

    def _setweights(self):
        "Apply dropout to the raw weights."
        for i, layer in enumerate(self.layer_names):
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(
                raw_w, p=self.weight_p, training=self.training)
            if self.idxs is not None:
                self.module._flat_weights[
                    self.idxs[i]] = self.module._parameters[layer]

    def forward(self, *args: ArgStar):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(
                raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'):
            self.module.reset()

    def _do_nothing(self): pass


class EmbeddingDropout(nn.Module):
    """Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    Credit: https://github.com/fastai/fastai/blob/54a9e3cf4fd0fa11fc2453a5389cc9263f6f0d77/fastai/text/models/awd_lstm.py#L64
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words: torch.LongTensor, scale: Optional[float] = None) -> torch.Tensor:
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1) # drop the entire word/wordpiece/character
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale:
            masked_embed.mul_(scale)
        return F.embedding(
            words, masked_embed, self.pad_idx, self.emb.max_norm,
            self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse
        )
