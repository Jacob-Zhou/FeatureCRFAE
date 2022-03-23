import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureScorer(nn.Module):

    def __init__(self, args, features=None):
        super(FeatureScorer, self).__init__()
        self.args = args
        self.features = features
        if "without_feature" not in args.keys():
            args.without_feature = False
        self.use_feature = (not args.without_feature) and (features
                                                           is not None)

        if self.use_feature:
            self.word_feature_weight = nn.Embedding(self.args.n_word_features,
                                                    self.args.n_labels)
            self.unigram_feature_weight = nn.Embedding(
                self.args.n_unigram_features, self.args.n_labels)
            self.bigram_feature_weight = nn.Embedding(
                self.args.n_bigram_features, self.args.n_labels)
            self.trigram_feature_weight = nn.Embedding(
                self.args.n_trigram_features, self.args.n_labels)
            self.morphology_feature_weight = nn.Parameter(
                torch.ones(3, self.args.n_labels))
        else:
            self.word_feature_weight = nn.Embedding(self.args.n_words,
                                                    self.args.n_labels)

    def extra_repr(self):
        s = f"use_feature={self.use_feature}, "
        if self.use_feature:
            s += "\n(morphology_feature_weight): "
            s += f"{self.morphology_feature_weight.__class__.__name__}("
            s += f"{', '.join(map(str, self.morphology_feature_weight.shape))})"
        return s

    def forward(self, words):
        """

        Args:
            words (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        if self.use_feature:
            (
                word_features,
                unigram_features,
                bigram_features,
                trigram_features,
                morphology_features,
            ) = self.features
            word_scores = self.word_feature_weight(word_features)
            unigram_scores = self.unigram_feature_weight(unigram_features)
            bigram_scores = self.bigram_feature_weight(bigram_features)
            trigram_scores = self.trigram_feature_weight(trigram_features)
            morphology_scores = (
                morphology_features.float() @ self.morphology_feature_weight)
            # [n_words, n_labels]
            scores = (word_scores + unigram_scores + bigram_scores +
                      trigram_scores + morphology_scores)
        else:
            scores = self.word_feature_weight.weight

        emits = F.embedding(words, torch.log_softmax(scores, dim=0))

        return emits


class GaussianScorer(nn.Module):

    def __init__(self, args, n_embed):
        super(GaussianScorer, self).__init__()
        self.args = args
        self.n_embed = n_embed
        self.n_labels = self.args.n_labels
        self.means = nn.Parameter(torch.Tensor(self.n_labels, self.n_embed))
        self.var = nn.Parameter(torch.ones(self.n_embed), requires_grad=False)
        self.factor = -self.n_embed / 2.0 * (math.log(2) + math.log(np.pi))
        self.log_density = self.factor - 0.5 * self.n_embed

    def init_params(self, means, var):
        noisy_means = torch.zeros_like(self.means.data)
        noisy_means.normal_().mul_(0.04)
        noisy_means = noisy_means + means.unsqueeze(0)
        self.means.data.copy_(means)
        self.var.data.copy_(var)
        # update log_density
        self.log_density = self.factor - 0.5 * self.var.log().sum()

    def extra_repr(self):
        s = "(means): "
        s += f"{self.means.__class__.__name__}("
        s += f"{', '.join(map(str, self.means.shape))})"
        s += "\n(var): "
        s += f"{self.var.__class__.__name__}("
        s += f"{', '.join(map(str, self.var.shape))})"
        return s

    def forward(self, embeddings):
        """

        Args:
            words (torch.Tensor): [batch_size, seq_len]

        Returns:

        """
        batch_size, seq_len, n_embed = embeddings.shape
        ep_size = torch.Size([batch_size, seq_len, self.n_labels, n_embed])
        embeddings = embeddings.unsqueeze(dim=2).expand(ep_size)
        means = self.means.expand(ep_size)
        var = self.var.expand(ep_size)

        return self.log_density - 0.5 * (
            (means - embeddings)**2 / var).sum(dim=-1)
