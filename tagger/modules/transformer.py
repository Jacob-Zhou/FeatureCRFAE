import torch
import torch.nn as nn

from tagger.utils.fn import pad
from .scalar_mix import ScalarMix

GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gpt2": "https://huggingface.co/gpt2/resolve/main/config.json",
    "gpt2-medium":
    "https://huggingface.co/gpt2-medium/resolve/main/config.json",
    "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/config.json",
    "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/config.json",
    "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/config.json",
}


class Transformer(nn.Module):
    def __init__(self,
                 model,
                 n_layers=0,
                 n_out=0,
                 stride=256,
                 pooling='mean',
                 pad_index=0,
                 dropout=0,
                 requires_grad=False):
        super().__init__()

        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.bert = AutoModel.from_pretrained(
            model,
            config=AutoConfig.from_pretrained(model,
                                              output_hidden_states=True))
        self.bert = self.bert.requires_grad_(requires_grad)

        self.model = model
        self.n_layers = n_layers or self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.stride = stride
        self.pooling = pooling
        self.pad_index = pad_index
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.max_len = int(
            max(0, self.bert.config.max_position_embeddings) or 1e12) - 2

        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.scalar_mix = ScalarMix(self.n_layers, dropout)
        self.projection = nn.Linear(
            self.hidden_size, self.n_out,
            False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.model}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pooling={self.pooling}, pad_index={self.pad_index}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        mask = subwords.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_subwords]
        subwords = pad(subwords[mask].split(lens.tolist()),
                       self.pad_index,
                       padding_side=self.tokenizer.padding_side)
        bert_mask = pad(mask[mask].split(lens.tolist()),
                        0,
                        padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        bert = self.bert(
            subwords[:, :self.max_len],
            attention_mask=bert_mask[:, :self.max_len].float())[-1]
        # [n_layers, batch_size, max_len, hidden_size]
        bert = bert[-self.n_layers:]
        # [batch_size, max_len, hidden_size]
        bert = self.scalar_mix(bert)
        # [batch_size, n_subwords, hidden_size]
        for i in range(self.stride,
                       (subwords.shape[1] - self.max_len + self.stride - 1) //
                       self.stride * self.stride + 1, self.stride):
            part = self.bert(
                subwords[:, i:i + self.max_len],
                attention_mask=bert_mask[:, i:i + self.max_len].float())[-1]
            bert = torch.cat((bert, self.scalar_mix(
                part[-self.n_layers:])[:, self.max_len - self.stride:]), 1)

        # [batch_size, seq_len]
        bert_lens = mask.sum(-1)
        bert_lens = bert_lens.masked_fill_(bert_lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        embed = bert.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(
            mask.unsqueeze(-1), bert[bert_mask])
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            embed = embed[:, :, 0]
        elif self.pooling == 'last':
            embed = embed.gather(2, (bert_lens - 1).unsqueeze(-1).repeat(
                1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        else:
            embed = embed.sum(2) / bert_lens.unsqueeze(-1)
        embed = self.projection(embed)

        # minus
        if 'gpt' in self.model:
            minus = embed[:, 1:] - embed[:, :-1]
            embed = torch.cat([embed[:, :1], minus], dim=1)
        else:
            embed = embed[:, 1:-1] - embed[:, :-2]

        return embed
