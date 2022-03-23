import copy
import os
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.nn as nn
from tagger.builders.builder import Builder
from tagger.models import FeatureHMM, VanillaHMM, GaussianHMM
from tagger.utils.common import bos, eos, pad, unk
from tagger.utils.config import Config
from tagger.utils.data import CoNLL, Dataset, FeatureField, Field
from tagger.utils.data import embedding
from tagger.utils.data.embedding import Embedding
from tagger.utils.data.field import ElmoField
from tagger.utils.fn import get_masks, heatmap, preprocess_hmm_fn
from tagger.utils.logging import get_logger, init_logger, progress_bar
from tagger.utils.metric import UnsupervisedPOSMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

logger = get_logger(__name__)


class HMMBuilder(Builder):
    NAME = "hmm"
    MODEL = VanillaHMM
    FILE_NAME = "hmm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FEAT, self.MASK = self.fields.FORM
        self.LABEL = self.fields.CPOS

    @classmethod
    def build(cls, path, **kwargs):
        args = Config(**locals())
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ud_mode = args.ud_mode
        ud_feature = args.ud_feature
        replace_punct = ud_feature or ud_feature
        args.update({"replace_punct": replace_punct})

        if os.path.exists(os.path.join(path,
                                       cls.FILE_NAME)) and not args.build:
            builder = cls.load(**args)
            builder.model = cls.MODEL(**args).to(args.device)
            return builder

        # fields
        # word field
        logger.info("Building the fields")

        # reconstruct field
        FEAT = Field(
            "froms",
            pad=pad,
            unk=unk,
            fn=partial(
                preprocess_hmm_fn,
                language=args.language,
                language_specific_strip=args.language_specific_strip,
                replace_punct=replace_punct,
            ),
            lower=args.ignore_capitalized,
        )

        # label field
        LABEL = Field("labels")
        MASK = Field("masks", use_vocab=False, fn=get_masks)

        fields = CoNLL(FORM=(FEAT, MASK), CPOS=LABEL)

        # load dataset
        train = Dataset(fields, args.train)

        # build vocab
        FEAT.build(train, args.min_freq)
        LABEL.build(train)

        if ud_mode:
            n_labels = 12
        else:
            n_labels = 45

        args.update({
            "n_words": FEAT.vocab.n_init,
            "n_labels": n_labels,
            "pad_index": FEAT.pad_index,
            "unk_index": FEAT.unk_index,
        })

        hmm = cls.MODEL().to(args.device)
        return cls(args, hmm, fields)

    def init_params(self, loader):
        return self

    def train(self, verbose=True, **kwargs):
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.fields.train()
        logger.info("Loading the data")

        train = Dataset(self.fields, args.train)
        evaluate = Dataset(self.fields, args.evaluate)
        test = Dataset(self.fields, args.test) if args.test else None

        # set the data loaders
        train.build(args.batch_size,
                    n_buckets=args.n_buckets,
                    shuffle=True,
                    seed=self.args.seed)
        evaluate.build(args.batch_size, n_buckets=args.n_buckets)
        logger.info(f"Train Dateset {train}")
        if args.evaluate != args.train:
            logger.info(f"Dev   Dateset {evaluate}")
        if test:
            test.build(args.batch_size, n_buckets=args.n_buckets)
            logger.info(f"Test  Dateset {test}")

        logger.info(f"{self.model}\n")
        total_time = timedelta()

        # Init parameters
        self.init_params(train.loader)

        min_loss, min_epoch = float("inf"), 0
        # optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            args.lr,
            (args.mu, args.nu),
            args.epsilon,
            args.weight_decay,
        )
        if args.decay_epochs > 0:
            # scheduler
            decay_steps = args.decay_epochs * len(train.loader)
            self.scheduler = ExponentialLR(self.optimizer,
                                           args.decay**(1 / decay_steps))
        else:
            self.scheduler = None

        for epoch in range(1, args.epochs + 1):
            logger.info(f"Epoch {epoch} / {args.epochs}:")
            start = datetime.now()
            # train
            self._train(train.loader)
            # evaluate
            dev_loss, dev_metric = self._evaluate(evaluate.loader)
            logger.info(f"{'dev:':10} Loss: {dev_loss:>8.4f} {dev_metric}")
            if test:
                test_loss, test_metric = self._evaluate(test.loader)
                dev_m2o_match, dev_o2o_match = dev_metric.match
                test_metric.set_match(dev_m2o_match, dev_o2o_match)
                logger.info(
                    f"{'test:':10} Loss: {test_loss:>8.4f} {test_metric}")
            heatmap(
                dev_metric.clusters.cpu(),
                list(self.LABEL.vocab.stoi.keys()),
                os.path.join(
                    self.args.path,
                    f"dev.{self.FILE_NAME}-{self.args.timestamp}.clusters",
                ),
                match=dev_metric.match[-1],
            )
            time_spent = datetime.now() - start
            total_time += time_spent
            # save the model if it is the best so far
            if dev_loss < min_loss:
                min_loss = dev_loss
                min_epoch = epoch
                self.save(args.path)
                logger.info(f"{time_spent}s elapsed (saved)\n")
            else:
                logger.info(f"{time_spent}s elapsed\n")

        logger.info(
            f"min_loss of {self.NAME} is {min_loss:.2f} at epoch {min_epoch}")

        # load best feature hmm model
        saved = self.load(args.path)

        dev_loss, dev_metric = saved._evaluate(evaluate.loader)
        if test:
            test_loss, test_metric = saved._evaluate(test.loader)
            dev_m2o_match, dev_o2o_match = dev_metric.match
            test_metric.set_match(dev_m2o_match, dev_o2o_match)
        heatmap(
            dev_metric.clusters.cpu(),
            list(self.LABEL.vocab.stoi.keys()),
            os.path.join(
                args.path,
                f"dev.{self.FILE_NAME}-{self.args.timestamp}.clusters"),
            match=dev_metric.match[-1],
        )
        logger.info(f"Epoch {min_epoch} saved")
        logger.info(f"{'dev:':10} Loss: {dev_loss:>8.4f} {dev_metric}")
        if test:
            logger.info(f"{'test:':10} Loss: {test_loss:>8.4f} {test_metric}")

        preds = saved._predict(evaluate.loader)
        for name, value in preds.items():
            setattr(evaluate, name, value)
        pred_path = os.path.join(
            args.path,
            f"dev.{self.FILE_NAME}-{self.args.timestamp}.predict.conllx")
        self.fields.save(pred_path, evaluate.sentences)
        logger.info(f"Saving predicted results to {pred_path}")
        logger.info(f"{total_time}s elapsed\n")

    def _train(self, loader):
        self.model.train()

        bar = progress_bar(loader)
        for feats, mask, _ in bar:
            mask = mask.bool()
            self.optimizer.zero_grad()
            emits, start, transitions, end = self.model(feats)
            # compute loss
            loss = self.model.loss(emits, start, transitions, end, mask)
            loss.backward()
            #
            if self.args.clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
                bar.set_postfix_str(
                    f" lr: {self.scheduler.get_last_lr()[0]:.4e}, loss: {loss.item():>8.4f}"
                )
            else:
                bar.set_postfix_str(
                    f" lr: {self.args.lr:.4e}, loss: {loss.item():>8.4f}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        metric = UnsupervisedPOSMetric(self.args.n_labels, self.args.device)

        total_loss = 0
        sent_count = 0
        for feats, mask, labels in loader:
            mask = mask.bool()
            sent_count += len(feats)
            emits, start, transitions, end = self.model(feats)
            # compute loss
            loss = self.model.loss(emits, start, transitions, end, mask)
            # predict
            predicts = self.model.decode(emits, start, transitions, end, mask)
            metric(predicts=predicts[mask], golds=labels[mask])
            total_loss += loss.item()
        total_loss /= sent_count
        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        labels = []
        for feats, mask, _ in progress_bar(loader):
            mask = mask.bool()
            lens = mask.sum(1).tolist()
            # ignore the first token of each sentence
            emits, start, transitions, end = self.model(feats)
            predicts = self.model.decode(emits, start, transitions, end, mask)
            labels.extend(predicts[mask].split(lens))

        labels = [[f"#C{t}#" for t in seq.tolist()] for seq in labels]
        preds = {"labels": labels}

        return preds


class FeatureHMMBuilder(HMMBuilder):
    NAME = "feature-hmm"
    MODEL = FeatureHMM
    FILE_NAME = "hmm"

    @classmethod
    def build(cls, path, **kwargs):
        args = Config(**locals())
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ud_mode = args.ud_mode
        ud_feature = args.ud_feature
        replace_punct = ud_feature or ud_feature
        args.update({"replace_punct": replace_punct})

        if os.path.exists(os.path.join(path,
                                       cls.FILE_NAME)) and not args.build:
            builder = cls.load(**args)
            builder.model = cls.MODEL(
                features=None if args.without_feature else FEAT.features,
                **args).to(args.device)
            return builder

        # fields
        # word field
        logger.info("Building the fields")

        # reconstruct field
        if args.without_feature:
            FEAT = Field(
                "feats",
                pad=pad,
                unk=unk,
                fn=partial(
                    preprocess_hmm_fn,
                    language=args.language,
                    language_specific_strip=args.language_specific_strip,
                    replace_punct=replace_punct,
                ),
                lower=args.ignore_capitalized,
            )
        else:
            FEAT = FeatureField(
                "feats",
                pad=pad,
                unk=unk,
                replace_mode=args.language
                if args.language_specific_strip else None,
                replace_punct=replace_punct,
                ud_feature_templates=ud_feature,
                language_specific_strip=args.language_specific_strip,
                lower=args.ignore_capitalized,
            )

        # label field
        LABEL = Field("labels")
        MASK = Field("masks", use_vocab=False, fn=get_masks)

        fields = CoNLL(FORM=(FEAT, MASK), CPOS=LABEL)

        # load dataset
        train = Dataset(fields, args.train)

        # build vocab
        FEAT.build(train, args.feat_min_freq)
        LABEL.build(train)

        if ud_mode:
            n_labels = 12
        else:
            n_labels = 45

        args.update({
            "n_labels": n_labels,
            "pad_index": FEAT.pad_index,
            "unk_index": FEAT.unk_index,
        })

        if not args.without_feature:
            args.update({
                "n_word_features": FEAT.n_word_features,
                "n_unigram_features": FEAT.n_unigram_features,
                "n_bigram_features": FEAT.n_bigram_features,
                "n_trigram_features": FEAT.n_trigram_features,
                "n_morphology_features": FEAT.n_morphology_features,
            })

        hmm = cls.MODEL(
            features=None if args.without_feature else FEAT.features,
            **args).to(args.device)
        return cls(args, hmm, fields)


class GaussianHMMBuilder(HMMBuilder):
    NAME = "gaussian-hmm"
    MODEL = GaussianHMM
    FILE_NAME = "gaussian-hmm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FEAT = self.fields.FORM
        self.LABEL = self.fields.CPOS

    @classmethod
    def build(cls, path, **kwargs):
        args = Config(**locals())
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ud_mode = args.ud_mode

        if os.path.exists(os.path.join(path,
                                       cls.FILE_NAME)) and not args.build:
            builder = cls.load(**args)
            builder.model = cls.MODEL(**args).to(args.device)
            return builder

        # fields
        # word field
        logger.info("Building the fields")

        # reconstruct field
        elmo_embedding = args.embed is None
        if elmo_embedding:
            WORD = ElmoField("words",
                             backbone="HIT" if ud_mode else "AllenNLP")
        else:
            WORD = Field("words",
                         pad=pad,
                         unk=unk,
                         lower=args.ignore_capitalized)

        MASK = Field("masks", use_vocab=False, fn=get_masks)

        # label field
        LABEL = Field("labels")

        fields = CoNLL(FORM=(WORD, MASK), CPOS=LABEL)

        # load dataset
        train = Dataset(fields, args.train)

        if not elmo_embedding:
            # build vocab
            WORD.build(
                train,
                args.min_freq,
                (Embedding.load(args.embed, unk="") if args.embed else None),
            )
            args.update({
                "n_words": WORD.vocab.n_init,
                "pad_index": WORD.pad_index,
                "unk_index": WORD.unk_index,
                "embed": WORD.embed
            })
        LABEL.build(train)

        if ud_mode:
            n_labels = 12
        else:
            n_labels = 45

        args.update({
            "n_labels": n_labels,
        })

        hmm = cls.MODEL(**args).to(args.device)
        return cls(args, hmm, fields)

    @torch.no_grad()
    def init_params(self, loader):
        logger.info("Init CRF-AE parameters")
        sample_batch = 3
        seq_len_set = set()
        embeddings = []
        for words, mask, _ in loader:
            if sample_batch <= 0:
                break
            seq_len = mask.shape[-1]
            if (seq_len > 5 and seq_len not in seq_len_set):
                seq_len_set.add(seq_len)
                sample_batch -= 1
                mask = mask.bool()
                output = self.model.encoder(words)
                embeddings.append(output[mask])
        embeddings = torch.cat(embeddings, dim=0)
        init_var, init_means = torch.var_mean(embeddings,
                                              dim=0,
                                              unbiased=False)
        self.model.init_params(means=init_means, var=init_var)
        return self
