import torch
from torch.nn.functional import one_hot
from sklearn.metrics.cluster import v_measure_score
from tagger.utils.alg import km_match


class Metric(object):
    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.0


class UnsupervisedPOSMetric(Metric):
    def __init__(self,
                 n_clusters,
                 device,
                 eps=1e-8,
                 default_score="many_to_one"):
        self.n_clusters = n_clusters
        self.eps = eps
        self.clusters = torch.zeros((self.n_clusters, self.n_clusters),
                                    device=device)
        self.pred = []
        self.gold = []
        self._match = None
        self.need_update = False
        self.use_own_match = True
        self.default_score = default_score or "many_to_one"

    def __call__(self, predicts, golds):
        """

        Args:
            predicts: [n]
            golds: [n]

        Returns:

        """
        self.need_update = True
        self.pred += predicts.tolist()
        self.gold += golds.tolist()
        predicts = one_hot(predicts, num_classes=self.n_clusters).unsqueeze(-1)
        golds = one_hot(golds, num_classes=self.n_clusters).unsqueeze(-2)
        clusters = (predicts * golds).sum(0)
        self.clusters += clusters

    def __repr__(self):
        return f"M-1: {self.many_to_one:.2%} " + \
            f"1-1: {self.one_to_one:.2%} " + \
            f"VM: {self.v_measure:.2%} "

    @property
    def match(self):
        self.calc_correct()
        return self._many_to_one_match, self._one_to_one_match

    @property
    def many_to_one(self):
        self.calc_correct()
        return float(self._many_to_one_correct /
                     (self.clusters.sum() + self.eps))

    @property
    def one_to_one(self):
        self.calc_correct()
        return float(self._one_to_one_correct /
                     (self.clusters.sum() + self.eps))

    @property
    def v_measure(self):
        return v_measure_score(self.gold, self.pred)

    def calc_correct(self):
        if self.need_update:
            if self.use_own_match or self._one_to_one_match is None:
                (self._one_to_one_correct,
                 self._one_to_one_match) = km_match(self.clusters)
            else:
                self._one_to_one_correct = self.clusters[
                    range(self.n_clusters), self._one_to_one_match].sum()

            if self.use_own_match or self._many_to_one_match is None:
                (_many_to_one_correct,
                 self._many_to_one_match) = self.clusters.max(dim=1)
                self._many_to_one_correct = _many_to_one_correct.sum()
            else:
                self._many_to_one_correct = self.clusters[
                    range(self.n_clusters), self._many_to_one_match].sum()
            self.need_update = False

    def set_match(self, many_to_one=None, one_to_one=None):
        self.use_own_match = False
        self.need_update = True
        self._one_to_one_match = one_to_one
        self._many_to_one_match = many_to_one

    @property
    def score(self):
        return self.get_score(self.default_score)

    def get_score(self, name="many_to_one"):
        if str.lower(name) in {"many_to_one", "M-1", "m-1", "m1"}:
            return self.many_to_one
        elif str.lower(name) in {"one_to_one", "1-1"}:
            return self.one_to_one
        elif str.lower(name) in {"v_measure", "vm"}:
            return self.v_measure
        else:
            raise NotImplementedError()