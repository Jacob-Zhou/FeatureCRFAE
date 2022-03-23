import os
from copy import copy
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.nn as nn
from tagger.builders.builder import Builder
from tagger.builders.hmm_builder import FeatureHMMBuilder, GaussianHMMBuilder
from tagger.models import CRFAE, FeatureHMM
from tagger.models.hmm import GaussianHMM
from tagger.utils.common import bos, eos, pad, unk
from tagger.utils.config import Config
from tagger.utils.data import (
    CoNLL,
    Dataset,
    ElmoField,
    Embedding,
    FeatureField,
    Field,
    SubwordField,
)
from tagger.utils.fn import get_masks, heatmap, preprocess_hmm_fn
from tagger.utils.logging import get_logger, init_logger, progress_bar
from tagger.utils.metric import Metric, UnsupervisedPOSMetric
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

logger = get_logger(__name__)


class CRFAEBuilder(Builder):
    NAME = "crf-ae"
    MODEL = CRFAE
    FILE_NAME = "crfae"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.FEAT, self.WORD, self.CHAR, self.MASK = self.fields.FORM
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
            builder.model = CRFAE(
                features=None if args.without_feature else FEAT.features,
                **args)
            if args.encoder == "lstm":
                builder.model.load_pretrained(builder.WORD.embed)
            builder.model.to(args.device)
            return builder

        # fields
        # word field
        logger.info("Building the fields")
        WORD = Field("words",
                     pad=pad,
                     unk=unk,
                     bos=bos,
                     eos=eos,
                     lower=args.ignore_capitalized)

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
        # char field
        if args.encoder == "elmo":
            CHAR = ElmoField("chars",
                             backbone="HIT" if ud_mode else "AllenNLP")
        elif args.encoder == "bert":
            from transformers import AutoTokenizer

            t = AutoTokenizer.from_pretrained(args.plm)
            CHAR = SubwordField(
                "chars",
                bos=t.cls_token,
                eos=t.sep_token,
                pad=t.pad_token,
                unk=t.unk_token,
                fix_len=args.fix_len,
                tokenize=t.tokenize,
            )
        elif args.encoder == "lstm":
            CHAR = SubwordField("chars",
                                pad=pad,
                                unk=unk,
                                bos=bos,
                                eos=eos,
                                fix_len=args.fix_len)
        else:
            raise NotImplementedError
        MASK = Field("masks", use_vocab=False, fn=get_masks)
        fields = CoNLL(FORM=(FEAT, WORD, CHAR, MASK), CPOS=LABEL)

        # load dataset
        train = Dataset(fields, args.train)

        # build vocab
        WORD.build(
            train,
            args.min_freq,
            (Embedding.load(args.embed, unk="") if args.embed else None),
        )
        FEAT.build(train, args.feat_min_freq)
        LABEL.build(train)
        if args.encoder == "bert":
            CHAR.vocab = t.get_vocab()
        elif args.encoder == "lstm":
            CHAR.build(train, args.min_freq)

        if ud_mode:
            n_labels = 12
        else:
            n_labels = 45

        args.update({
            "n_words": WORD.vocab.n_init,
            "n_chars": len(CHAR.vocab),
            "n_labels": n_labels,
            "pad_index": WORD.pad_index,
            "unk_index": WORD.unk_index,
        })

        if not args.without_feature:
            args.update({
                "n_word_features": FEAT.n_word_features,
                "n_unigram_features": FEAT.n_unigram_features,
                "n_bigram_features": FEAT.n_bigram_features,
                "n_trigram_features": FEAT.n_trigram_features,
                "n_morphology_features": FEAT.n_morphology_features,
            })
        else:
            args.update({
                "n_recons_word": FEAT.vocab.n_init,
            })
        crf_ae = CRFAE(
            features=None if args.without_feature else FEAT.features, **args)
        if args.encoder == "lstm":
            crf_ae.encoder.load_pretrained(WORD.embed)
        crf_ae = crf_ae.to(args.device)
        return cls(args, crf_ae, fields)

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

        total_time = timedelta()

        if args.init_model is not None:
            self.init_params(train, evaluate, test)
        train.reset_loader(args.batch_size, shuffle=True, seed=self.args.seed)

        logger.info(f"{self.model}\n")

        logger.info("Train CRF AE")
        params = [
            {
                "params": self.model.feature_hmm.parameters(),
                "lr": args.recons_lr
            },
            {
                "params": self.model.represent_ln.parameters()
            },
            {
                "params": self.model.encoder_emit_scorer.parameters()
            },
            {
                "params": self.model.encoder_emit_ln.parameters()
            },
            {
                "params": self.model.start
            },
            {
                "params": self.model.transitions
            },
            {
                "params": self.model.end
            },
        ]

        if args.init_model is not None and self.model.encoder.scalar_mix is not None:
            params.append(
                {"params": self.model.encoder.scalar_mix.parameters()})

        self.optimizer = Adam(params, args.encoder_lr, (args.mu, args.nu),
                              args.epsilon, args.weight_decay)

        # scheduler
        if args.decay_epochs > 0:
            # scheduler
            decay_steps = args.decay_epochs * len(train.loader)
            self.scheduler = ExponentialLR(self.optimizer,
                                           args.decay**(1 / decay_steps))
        else:
            self.scheduler = None

        best_e, best_metric = 1, Metric()
        min_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            logger.info(f"Epoch {epoch} / {args.epochs}:")
            start = datetime.now()
            # train
            self._train(train.loader)
            # evaluate
            dev_loss, dev_metric = self._evaluate(evaluate.loader)
            if test:
                test_loss, test_metric = self._evaluate(test.loader)
                dev_m2o_match, dev_o2o_match = dev_metric.match
                test_metric.set_match(dev_m2o_match, dev_o2o_match)
            heatmap(
                dev_metric.clusters.cpu(),
                list(self.LABEL.vocab.stoi.keys()),
                os.path.join(
                    self.args.path,
                    f"dev.{self.FILE_NAME}-{self.args.timestamp}.clusters",
                ),
                match=dev_metric.match[-1],
            )
            logger.info(f"{'dev:':10} Loss: {dev_loss:>8.4f} {dev_metric}")
            if test:
                logger.info(
                    f"{'test:':10} Loss: {test_loss:>8.4f} {test_metric}")

            time_spent = datetime.now() - start
            total_time += time_spent
            # save the model if it is the best so far
            if dev_loss < min_loss:
                best_e, best_metric, min_loss = epoch, dev_metric, dev_loss
                if test:
                    best_test_metric, min_test_loss = test_metric, test_loss
                self.save(args.path)
                logger.info(f"{time_spent}s elapsed (saved)\n")
            else:
                logger.info(f"{time_spent}s elapsed\n")

        logger.info(f"max score of CRF is at epoch {best_e}")
        logger.info(f"{'dev:':10} Loss: {min_loss:>8.4f} {best_metric}")
        if test:
            logger.info(
                f"{'test:':10} Loss: {min_test_loss:>8.4f} {best_test_metric}")
        saved = self.load(args.path)
        preds = saved._predict(evaluate.loader)
        for name, value in preds.items():
            setattr(evaluate, name, value)
        pred_path = os.path.join(
            args.path,
            f"dev.{self.FILE_NAME}-{self.args.timestamp}.predict.conllx")
        self.fields.save(pred_path, evaluate.sentences)
        logger.info(f"Saving predicted results to {pred_path}")
        logger.info(f"{total_time}s elapsed\n")

    def init_params(self, train, evaluate, test):
        logger.info(f"Initialize CRF-AE using {self.args.init_model} model")
        hmm_args = copy(self.args)
        hmm_args.update({
            "lr": self.args.hmm_lr,
            "mu": self.args.hmm_mu,
            "nu": self.args.hmm_nu,
            "epsilon": self.args.hmm_epsilon,
            "weight_decay": self.args.hmm_weight_decay,
            "clip": self.args.hmm_clip,
            "decay": self.args.hmm_decay,
            "decay_epochs": self.args.hmm_decay_epochs,
            "epochs": self.args.hmm_epochs,
        })
        if self.args.init_model == "feature-hmm":
            if self.args.hmm_path is None or not os.path.exists(
                    self.args.hmm_path):
                self.args.hmm_path = self.args.path
                hmm_args.update({
                    "path":
                    self.args.hmm_path,
                    "features":
                    None if self.args.without_feature else self.FEAT.features
                })
                feature_hmm = FeatureHMM(**hmm_args).to(self.args.device)
                hmm_fields = CoNLL(FORM=(self.FEAT, self.MASK),
                                   CPOS=self.LABEL)
                hmm_builder = FeatureHMMBuilder(hmm_args, feature_hmm,
                                                hmm_fields)
                hmm_builder.train(**hmm_args)
            teacher = FeatureHMMBuilder.load(self.args.hmm_path).model
            # load best feature hmm model
            self.model.feature_hmm = teacher
        elif self.args.init_model == "gaussian-hmm":
            if self.args.hmm_path is None or not os.path.exists(
                    self.args.hmm_path):
                self.args.hmm_path = self.args.path
                hmm_args.update({
                    "path":
                    self.args.hmm_path,
                    "n_embed": (self.args.n_embed if self.args.encoder
                                == "lstm" else self.args.n_pretrained),
                    "embed":
                    (self.WORD.embed if self.args.encoder == "lstm" else None)
                })
                gaussian_hmm = GaussianHMM(**hmm_args).to(self.args.device)
                if self.args.encoder == "lstm":
                    WORD = Field("words", pad=pad, unk=unk, lower=True)
                    WORD.vocab = self.WORD.vocab
                    WORD.embed = self.WORD.embed
                    hmm_fields = CoNLL(FORM=(WORD, self.MASK), CPOS=self.LABEL)
                elif self.args.encoder == "elmo":
                    hmm_fields = CoNLL(FORM=(self.CHAR, self.MASK),
                                       CPOS=self.LABEL)
                else:
                    raise NotImplementedError
                hmm_builder = GaussianHMMBuilder(hmm_args, gaussian_hmm,
                                                 hmm_fields)
                hmm_builder.train(**hmm_args)
            teacher = GaussianHMMBuilder.load(self.args.hmm_path).model
        else:
            raise NotImplementedError

        logger.info("Init CRF-AE parameters")
        # optimizer
        optimizer = Adam(
            self.model.parameters(),
            self.args.init_lr,
            (self.args.init_mu, self.args.init_nu),
            self.args.init_epsilon,
            self.args.init_weight_decay,
        )
        # scheduler
        if self.args.init_decay_epochs > 0:
            # scheduler
            decay_steps = self.args.init_decay_epochs * len(train.loader)
            scheduler = ExponentialLR(optimizer,
                                      self.args.init_decay**(1 / decay_steps))
        else:
            scheduler = None

        for epoch in range(1, self.args.init_epochs + 1):
            logger.info(f"Epoch {epoch} / {self.args.init_epochs}:")
            start = datetime.now()
            # train
            self._init_crf(train.loader, teacher, optimizer, scheduler)
            time_spent = datetime.now() - start
            logger.info(f"{time_spent}s elapsed\n")

        dev_loss, dev_metric = self._evaluate(evaluate.loader)
        if test:
            test_loss, test_metric = self._evaluate(test.loader)
            dev_m2o_match, dev_o2o_match = dev_metric.match
            test_metric.set_match(dev_m2o_match, dev_o2o_match)
        heatmap(
            dev_metric.clusters.cpu(),
            list(self.LABEL.vocab.stoi.keys()),
            os.path.join(
                self.args.path,
                f"dev.{self.FILE_NAME}-{self.args.timestamp}.init.clusters",
            ),
            match=dev_metric.match[-1],
        )
        logger.info(f"{'CRF init:':10}")
        logger.info(f"{'dev:':10} Loss: {dev_loss:>8.4f} {dev_metric}")
        if test:
            logger.info(f"{'test:':10} Loss: {test_loss:>8.4f} {test_metric}")

    def _init_crf(self, loader, teacher, optimizer, scheduler):
        self.model.train()
        bar = progress_bar(loader)
        for feats, words, chars, mask, _ in bar:
            mask = mask.bool()
            # use feature hmm to generate labels
            with torch.no_grad():
                if self.args.init_model == "feature-hmm":
                    inputs = feats
                elif self.args.init_model == "gaussian-hmm":
                    if self.args.encoder == "lstm":
                        inputs = words
                    else:
                        inputs = chars
                else:
                    raise NotImplementedError
                emits, start, transitions, end = teacher(inputs)
                labels = teacher.decode(emits, start, transitions, end, mask)

            optimizer.zero_grad()

            encoder_emits = self.model(words, chars)

            # compute loss
            if self.args.init_model == "feature-hmm":
                # do not tuning decoder part
                loss = self.model.crf_loss(None, encoder_emits, labels, mask)
            else:
                # init decoder part
                loss = self.model.crf_loss(feats, encoder_emits, labels, mask)

            loss.backward()
            #
            if self.args.init_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.args.init_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                bar.set_postfix_str(
                    f" lr: {scheduler.get_last_lr()[0]:.4e}, loss: {loss.item():>8.4f}"
                )
            else:
                bar.set_postfix_str(
                    f" lr: {self.args.init_lr:.4e}, loss: {loss.item():>8.4f}")

    def _train(self, loader):
        self.model.train()
        bar = progress_bar(loader)
        for feats, words, chars, mask, _ in bar:
            mask = mask.bool()
            self.optimizer.zero_grad()
            encoder_emits = self.model(words, chars)
            # compute loss
            loss = self.model.loss(feats, encoder_emits, mask)

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

        total_loss = 0
        metric = UnsupervisedPOSMetric(self.args.n_labels, self.args.device)
        sent_count = 0
        for feats, words, chars, mask, labels in loader:
            mask = mask.bool()
            sent_count += len(words)
            encoder_emits = self.model(words, chars)
            # compute loss
            loss = self.model.loss(feats, encoder_emits, mask)
            # predict
            predicts = self.model.decode(feats, encoder_emits, mask)
            metric(predicts=predicts[mask], golds=labels[mask])
            total_loss += loss.item()
        total_loss /= sent_count
        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        labels = []
        for feats, words, chars, mask, _ in progress_bar(loader):
            mask = mask.bool()
            lens = mask.sum(1).tolist()
            # ignore the first token of each sentence
            encoder_emits = self.model(words, chars)
            predicts = self.model.decode(feats, encoder_emits, mask)
            labels.extend(predicts[mask].split(lens))

        labels = [[f"#C{t}#" for t in seq.tolist()] for seq in labels]
        preds = {"labels": labels}

        return preds
