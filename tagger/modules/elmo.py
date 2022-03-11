import os
from glob import glob
from tagger.modules.scalar_mix import ScalarMix
from tagger.utils.fn import pad

import torch
import torch.nn as nn
from allennlp.common import FromParams
from allennlp.nn.util import remove_sentence_boundaries
from elmoformanylangs import Embedder as EFML


class Elmo(torch.nn.Module, FromParams):
    def __init__(self,
                 backbone,
                 model_path,
                 layer=None,
                 dropout=0.33,
                 requires_grad=False,
                 fd_repr=True):
        """

        Args:
            layer (int):
            dropout (float):
        """
        super().__init__()
        assert isinstance(backbone, str)
        self.backbone = str.lower(backbone)
        assert self.backbone in {"hit", "allennlp"}
        self.fd_repr = fd_repr
        self.layer = layer
        self.dropout = dropout
        self._dropout = nn.Dropout(p=dropout)
        self.requires_grad = requires_grad
        if self.backbone == "hit":
            self.elmo = EFML(model_dir=model_path)
        else:
            json_paths = glob(os.path.join(model_path, "*.json"))
            hdf5_paths = glob(os.path.join(model_path, "*.hdf5"))
            if len(json_paths) * len(hdf5_paths) > 1:
                raise RuntimeError("Elmo config is ambiguous")
            # it will effect the os.environ['CUDA_VISIBLE_DEVICES'] = device
            from allennlp.modules.elmo import _ElmoBiLm
            self.elmo = _ElmoBiLm(
                options_file=json_paths[0],
                weight_file=hdf5_paths[0],
                requires_grad=requires_grad,
            )
        if layer is None:
            self.scalar_mix = ScalarMix(n_layers=2)
        elif layer == 3:
            self.scalar_mix = ScalarMix(n_layers=3)
        else:
            self.scalar_mix = None

    def __repr__(self):
        s = self.__class__.__name__ + f"("
        s += f"layer={self.layer if self.layer is not None else 'all'}, "
        s += f"dropout={self.dropout}, "
        s += f"fd_repr={self.fd_repr}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"
        s += ')'

        return s

    def forward(self, chars):
        """

        Args:
            chars : `torch.Tensor`, required.
            Shape `(batch_size, timesteps, 50)` of character ids representing the current batch.
        Returns:

        """

        if self.backbone == "hit":
            # [B, 3, L, H]
            layer_activations = pad(self.elmo.sents2elmo(chars)).unbind(1)

            if self.layer is None:
                res = self.scalar_mix(layer_activations[-2:])
            elif self.layer == 3:
                res = self.scalar_mix(layer_activations)
            else:
                res = layer_activations[self.layer]
        else:
            bilm_output = self.elmo(chars)
            layer_activations = bilm_output["activations"]
            mask_with_bos_eos = bilm_output["mask"]

            if self.layer is None:
                representation_with_bos_eos = self.scalar_mix(
                    layer_activations[-2:])
            elif self.layer == 3:
                representation_with_bos_eos = self.scalar_mix(
                    layer_activations)
            else:
                representation_with_bos_eos = layer_activations[self.layer]
            res, _ = remove_sentence_boundaries(representation_with_bos_eos,
                                                mask_with_bos_eos)

        res = self._dropout(res)

        if self.fd_repr:
            # get minus
            forward, backward = torch.chunk(res, 2, dim=-1)

            forward_minus = forward[:, 1:] - forward[:, :-1]
            forward_minus = torch.cat([forward[:, :1], forward_minus], dim=1)
            backward_minus = backward[:, :-1] - backward[:, 1:]
            backward_minus = torch.cat([backward_minus, backward[:, -1:]],
                                       dim=1)
            # [batch_size, seq_len, n_elmo]
            res = torch.cat([forward_minus, backward_minus], dim=-1)
        return res
