import torch
import torch.nn as nn
from tagger.modules import FeatureScorer
from tagger.utils.config import Config


class FeatureHMM(nn.Module):
    def __init__(self, features=None, **kwargs):
        super(FeatureHMM, self).__init__()
        self.args = args = Config().update(locals())

        # features compute emits
        if args.without_feature or features is not None:
            self.features = features
        else:
            raise RuntimeError('Both features is None when we need features')

        self.feature_scorer = FeatureScorer(args, features)

        # t
        self.start = nn.Parameter(torch.randn(args.n_labels))
        self.end = nn.Parameter(torch.randn(args.n_labels))
        self.transitions = nn.Parameter(
            torch.randn(args.n_labels, args.n_labels))

    def extra_repr(self):
        s = f"(start): {self.start.__class__.__name__}({', '.join(map(str, self.start.shape))})\n"
        s += f"(end): {self.end.__class__.__name__}({', '.join(map(str, self.end.shape))})\n"
        s += f"(transitions): {self.transitions.__class__.__name__}({', '.join(map(str, self.transitions.shape))})"
        return s

    def forward(self, words):
        """

        Args:
            words (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        emits = self.feature_scorer(words)
        start = torch.log_softmax(self.start, dim=-1)
        transitions = torch.log_softmax(self.transitions, dim=-1)
        end = torch.log_softmax(self.end, dim=-1)
        return emits, start, transitions, end

    def loss(self, emits, start, transitions, end, mask):
        """

        Args:
            emits (torch.Tensor): [batch_size, seq_len, n_labels]
            start (torch.Tensor): [n_labels]
            transitions (torch.Tensor): [n_labels, n_labels]
            end (torch.Tensor): [n_labels]
            mask (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        batch_size, seq_len, n_labels = emits.shape

        # start
        # [1, n_labels] + [batch_size, n_labels]
        log_score = start.unsqueeze(0) + emits[:, 0]

        for i in range(1, seq_len):
            # [batch_size, n_labels, 1] + [1, n_labels, n_labels] + [batch_size, 1, n_labels]
            score = log_score.unsqueeze(-1) + transitions.unsqueeze(
                0) + emits[:, i].unsqueeze(1)
            log_score[mask[:, i]] = torch.logsumexp(score, dim=1)[mask[:, i]]

        # end
        log_p = torch.logsumexp(log_score + end.unsqueeze(0), dim=-1)
        return -log_p.sum()

    def decode(self, emits, start, transitions, end, mask):
        """

        Args:
            emits (torch.Tensor): [batch_size, seq_len, n_labels]
            start (torch.Tensor): [n_labels]
            transitions (torch.Tensor): [n_labels, n_labels]
            end (torch.Tensor): [n_labels]
            mask (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        batch_size, seq_len, n_labels = emits.shape

        last_next_pos = mask.sum(1)

        # [batch_size, seq_len + 1, n_labels]
        path = emits.new_zeros((batch_size, seq_len + 1, n_labels)).long()

        # start
        # [batch_size, n_labels]
        score = start.unsqueeze(0) + emits[:, 0]

        for i in range(1, seq_len):
            # [batch_size, n_labels, 1] + [batch_size, n_labels, n_labels] => [batch_size, n_labels, n_labels]
            temp_score = score.unsqueeze(-1) + transitions.unsqueeze(
                0) + emits[:, i].unsqueeze(1)
            # [batch_size, n_labels]
            temp_score, path[:, i] = torch.max(temp_score, dim=1)
            score[mask[:, i]] = temp_score[mask[:, i]]
            path[:, i][~mask[:, i]] = 0

        # end
        score = score + end.unsqueeze(0)

        batch = torch.arange(batch_size, dtype=torch.long).to(emits.device)
        path[batch, last_next_pos, 0] = torch.argmax(score, dim=-1)

        # tags: [batch_size, seq_len]
        tags = emits.new_zeros((batch_size, seq_len)).long()
        # pre_tags: [batch_size, 1]
        pre_tags = emits.new_zeros((batch_size, 1)).long()
        for i in range(seq_len, 0, -1):
            j = i - seq_len - 1
            # pre_tags: [batch_size, 1]
            pre_tags = torch.gather(path[:, i], 1, pre_tags)
            tags[:, j] = pre_tags.squeeze()
        return tags
