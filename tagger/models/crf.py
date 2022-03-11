import torch
import torch.nn as nn
from tagger.models.feature_hmm import FeatureHMM
from tagger.modules import Elmo, LSTMEncoder, Transformer
from tagger.utils.config import Config


class CRFAE(nn.Module):
    def __init__(self, feature_hmm=None, features=None, **kwargs):
        """
        use ELMo, CRF auto encoder and feature HMM

        Args:
            args:
        """
        super(CRFAE, self).__init__()
        self.args = args = Config().update(locals())

        if self.args.encoder == "elmo":
            self.encoder = Elmo(layer=args.layer,
                                dropout=args.dropout,
                                backbone="HIT" if args.ud_mode else "AllenNLP",
                                model_path=args.plm,
                                fd_repr=not args.without_fd_repr)
        elif self.args.encoder == "bert":
            self.encoder = Transformer(model=args.plm,
                                       n_layers=args.n_layers,
                                       dropout=args.dropout)
            self.args.update({"n_pretrained": self.encoder.hidden_size})
        elif self.args.encoder == "lstm":
            self.encoder = LSTMEncoder(
                n_words=args.n_words,
                n_embed=args.n_embed,
                n_chars=args.n_chars,
                n_char_embed=args.n_char_embed,
                n_char_lstm_embed=args.n_char_lstm_embed,
                n_layers=args.n_layers,
                n_hidden=args.n_hidden,
                pad_index=args.pad_index,
                embed_dropout=args.embed_dropout,
                dropout=args.dropout,
            )
            self.args.update({"n_pretrained": 2 * args.n_hidden})
        else:
            raise RuntimeError

        # encoder
        self.represent_ln = nn.LayerNorm(args.n_pretrained)
        if args.n_bottleneck <= 0:
            self.encoder_emit_scorer = nn.Linear(args.n_pretrained,
                                                 args.n_labels)
        else:
            self.encoder_emit_scorer = nn.Sequential(
                nn.Linear(args.n_pretrained, args.n_bottleneck),
                nn.LeakyReLU(), nn.Linear(args.n_bottleneck, args.n_labels))
        self.encoder_emit_ln = nn.LayerNorm(args.n_labels)
        self.start = nn.Parameter(torch.randn((args.n_labels, )))
        self.transitions = nn.Parameter(
            torch.randn((args.n_labels, args.n_labels)))
        self.end = nn.Parameter(torch.randn((args.n_labels, )))

        # decoder
        if feature_hmm is None:
            if args.without_feature or features is not None:
                self.feature_hmm = FeatureHMM(**args)
            else:
                raise RuntimeError('Both feature_hmm and features is None')
        else:
            self.feature_hmm = feature_hmm

    def extra_repr(self):
        s = f"(start): {self.start.__class__.__name__}({', '.join(map(str, self.start.shape))})\n"
        s += f"(end): {self.end.__class__.__name__}({', '.join(map(str, self.end.shape))})\n"
        s += f"(transitions): {self.transitions.__class__.__name__}({', '.join(map(str, self.transitions.shape))})"
        return s

    def forward(self, words, chars):
        """

        Args:
            inputs:

        Returns:

        """

        # [batch_size, seq_len, n_elmo]
        if self.args.encoder in {"elmo", 'bert'}:
            represent = self.encoder(chars)
        else:
            represent = self.encoder(words, chars)

        # 发射分值
        # [batch_size, seq_len, n_labels]
        represent = self.represent_ln(represent)
        encoder_emits = self.encoder_emit_scorer(represent)
        encoder_emits = self.encoder_emit_ln(encoder_emits)

        return encoder_emits

    def loss(self, words, encoder_emits, mask):
        """

        Args:
            words:
            encoder_emits:
            mask:

        Returns:

        """
        _, seq_len, _ = encoder_emits.shape

        encoder_emits = encoder_emits.double()
        decoder_emits = self.feature_hmm.feature_scorer(words).double()

        start = self.start.double()
        transitions = self.transitions.double()
        end = self.end.double()

        # start
        # [1, n_labels] + [batch_size, n_labels]
        log_alpha = start.unsqueeze(0) + encoder_emits[:, 0]
        # [batch_size, n_labels]
        log_beta = log_alpha + decoder_emits[:, 0]

        for i in range(1, seq_len):
            # [batch_size, 1, n_labels] + [1, n_labels, n_labels]
            crf_scores = encoder_emits[:, i].unsqueeze(
                1) + transitions.unsqueeze(0)
            # [batch_size, n_labels, 1] +  [batch_size, n_labels, n_labels]
            alpha_scores = log_alpha.unsqueeze(-1) + crf_scores
            # [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels] + [batch_size, 1, n_labels]
            beta_scores = log_beta.unsqueeze(
                -1) + crf_scores + decoder_emits[:, i].unsqueeze(1)

            log_alpha[mask[:, i]] = torch.logsumexp(alpha_scores,
                                                    dim=1)[mask[:, i]]
            log_beta[mask[:, i]] = torch.logsumexp(beta_scores, dim=1)[mask[:,
                                                                            i]]

        # end
        # [batch_size, n_labels] + [1, n_labels]
        alpha_scores = log_alpha + end.unsqueeze(0)
        # [batch_size, n_labels] + [1, n_labels]
        beta_scores = log_beta + end.unsqueeze(0)

        # [batch_size]
        log_alpha = torch.logsumexp(alpha_scores, dim=-1)
        log_beta = torch.logsumexp(beta_scores, dim=-1)

        return (log_alpha - log_beta).sum().float()

    def crf_loss(self, encoder_emits, labels, mask):
        """
        compute crf loss to train encoder

        Args:
            encoder_emits (torch.Tensor): [batch_size, seq_len, n_labels]
            labels (torch.Tensor): [batch_size, seq_len]
            mask (torch.Tensor): [batch_size, seq_len]

        Returns:

        """

        encoder_emits = encoder_emits.double()

        start = self.start.double()
        transitions = self.transitions.double()
        end = self.end.double()

        # compute log p
        batch_size, seq_len, n_labels = encoder_emits.shape
        # [1, n_labels] + [batch_size, n_labels]
        log_score = start.unsqueeze(0) + encoder_emits[:, 0]
        for i in range(1, seq_len):
            # [batch_size, n_labels, 1] + [1, n_labels, n_labels] + [batch_size, 1, n_labels]
            score = log_score.unsqueeze(-1) + transitions.unsqueeze(
                0) + encoder_emits[:, i].unsqueeze(1)
            log_score[mask[:, i]] = torch.logsumexp(score, dim=1)[mask[:, i]]
        log_p = torch.logsumexp(log_score + end.unsqueeze(0), dim=-1).sum()

        # compute score for pseudo labels
        batch = torch.arange(batch_size).to(encoder_emits.device)
        last_pos = mask.sum(-1) - 1
        # [batch_size]
        score = (start[labels[:, 0]] + end[labels[batch, last_pos]]).sum()
        # emits score
        score += torch.gather(encoder_emits[mask],
                              dim=-1,
                              index=labels[mask].unsqueeze(-1)).sum()
        # transitions score
        for i in range(1, seq_len):
            score += transitions[labels[:, i - 1][mask[:, i]],
                                 labels[:, i][mask[:, i]]].sum()
        return (log_p - score).float()

    def decode(self, words, encoder_emits, mask):
        """

        Args:
            words (torch.Tensor): [batch_size, seq_len]
            encoder_emits (torch.Tensor): [batch_size, seq_len, n_labels]
            mask (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        batch_size, seq_len, n_labels = encoder_emits.shape

        decoder_emits = self.feature_hmm.feature_scorer(words)

        start_transitions = self.start
        transitions = self.transitions
        end_transitions = self.end

        last_next_position = mask.sum(1)

        # [batch_size, seq_len + 1, n_labels]
        path = encoder_emits.new_zeros(
            (batch_size, seq_len + 1, n_labels)).long()

        # start
        # [batch_size, n_labels]
        score = start_transitions.unsqueeze(
            0) + encoder_emits[:, 0] + decoder_emits[:, 0]

        for i in range(1, seq_len):
            # [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels] => [batch_size, n_labels, n_labels]
            temp_score = score.unsqueeze(-1) + transitions.unsqueeze(
                0) + encoder_emits[:, i].unsqueeze(
                    1) + decoder_emits[:, i].unsqueeze(1)
            # [batch_size, n_labels]
            temp_score, path[:, i] = torch.max(temp_score, dim=1)
            score[mask[:, i]] = temp_score[mask[:, i]]
            path[:, i][~mask[:, i]] = 0

        # end
        score = score + end_transitions.unsqueeze(0)

        # 将每一个句子有效末尾后面一个位置的<pad>位指向的tag改为delta中记录的最大tag
        batch = torch.arange(batch_size,
                             dtype=torch.long).to(encoder_emits.device)
        path[batch, last_next_position, 0] = torch.argmax(score, dim=-1)

        # tags: [batch_size, seq_len]
        tags = encoder_emits.new_zeros((batch_size, seq_len)).long()
        # pre_tags: [batch_size, 1]
        pre_tags = encoder_emits.new_zeros((batch_size, 1)).long()
        for i in range(seq_len, 0, -1):
            j = i - seq_len - 1
            # pre_tags: [batch_size, 1]
            pre_tags = torch.gather(path[:, i], 1, pre_tags)
            tags[:, j] = pre_tags.squeeze()
        return tags
