from tagger.modules import BiLSTM, CharLSTM, IndependentDropout, SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    def __init__(self,
                 n_words,
                 n_embed,
                 n_chars,
                 n_char_embed,
                 n_char_lstm_embed,
                 n_layers,
                 n_hidden,
                 pad_index=0,
                 embed_dropout=0,
                 dropout=0):
        super().__init__()
        self.pad_index = pad_index
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        self.feat_embed = CharLSTM(n_chars=n_chars,
                                   n_embed=n_char_embed,
                                   n_out=n_char_lstm_embed,
                                   pad_index=pad_index)
        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_embed + n_char_lstm_embed,
                           hidden_size=n_hidden,
                           num_layers=n_layers,
                           dropout=dropout)
        self.lstm_dropout = SharedDropout(p=dropout)

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, words, chars):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        # [batch_size, seq_len, n_elmo]
        _, seq_len = words.shape
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)

        char_embed = self.feat_embed(chars)
        word_embed, char_embed = self.embed_dropout(word_embed, char_embed)
        embed = torch.cat((word_embed, char_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        forward, backward = torch.chunk(x, 2, dim=-1)

        forward_minus = forward[:, 1:-1] - forward[:, :-2]
        backward_minus = backward[:, 1:-1] - backward[:, 2:]

        # [batch_size, seq_len, n_elmo]
        represent = torch.cat([forward_minus, backward_minus], dim=-1)
        return represent
