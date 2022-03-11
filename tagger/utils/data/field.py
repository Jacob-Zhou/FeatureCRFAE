import re
from collections import Counter
from tagger.utils.fn import pad, replace_digit_fn, replace_punct_fn, replace_punct, ispunct, strip_word

import torch

from .vocab import Vocab


class RawField(object):
    def __init__(self, name, fn=None):
        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        return self.fn(sequence) if self.fn is not None else sequence

    def transform(self, sequences):
        return [self.preprocess(seq) for seq in sequences]

    def compose(self, sequences):
        return sequences


class Field(RawField):
    def __init__(self,
                 name,
                 pad=None,
                 unk=None,
                 bos=None,
                 eos=None,
                 lower=False,
                 use_vocab=True,
                 tokenize=None,
                 fn=None,
                 is_placeholder=False):
        super(Field, self).__init__(name, fn)
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos
        self.lower = lower
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.is_placeholder = is_placeholder

        self.specials = [
            token for token in [pad, unk, bos, eos] if token is not None
        ]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        s += ", ".join(params)
        s += ")"

        return s

    @property
    def pad_index(self):
        if self.pad is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.pad]
        return self.specials.index(self.pad)

    @property
    def unk_index(self):
        if self.unk is None:
            return 0
        if hasattr(self, 'vocab'):
            return self.vocab[self.unk]
        return self.specials.index(self.unk)

    @property
    def bos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.bos]
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        if hasattr(self, 'vocab'):
            return self.vocab[self.eos]
        return self.specials.index(self.eos)

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def preprocess(self, sequence):
        """
        Load a single example using this field, tokenizing if necessary.
        The sequence will be first passed to `self.fn` if available.
        If `self.tokenize` is not None, the input will be tokenized.
        Then the input will be optionally lowercased.

        Args (List):
            The sequence to be preprocessed.

        Returns:
            sequence (List):
                the preprocessed sequence.
        """

        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence)
        if self.lower:
            sequence = [str.lower(token) for token in sequence]

        return sequence

    def build(self, dataset, min_freq=1, embed=None):
        """
        Construct the Vocab object for this field from the dataset.
        If the Vocab has already existed, this function will have no effect.

        Args:
            dataset (Dataset):
                A Dataset instance. One of the attributes should be named after the name of this field.
            min_freq (int, default: 1):
                The minimum frequency needed to include a token in the vocabulary.
            embed (Embedding, default: None):
                An Embedding instance, words in which will be extended to the vocabulary.
        """
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(dataset, self.name)
        counter = Counter(token for seq in sequences
                          for token in self.preprocess(seq))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab.stoi), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors
            self.embed /= torch.std(self.embed)

    def transform(self, sequences):
        """
        Turns a list of sequences that use this field into tensors.

        Each sequence is first preprocessed and then numericalized if needed.

        Args:
            sequences (List[List[str]]):
                A List of sequences.

        Returns:
            sequences (List[Tensor]):
                A list of tensors transformed from the input sequences.
        """

        sequences = [self.preprocess(seq) for seq in sequences]
        if self.use_vocab:
            sequences = [self.vocab[seq] for seq in sequences]
        if self.bos:
            sequences = [[self.bos_index] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [self.eos_index] for seq in sequences]
        sequences = [torch.tensor(seq) for seq in sequences]

        return sequences

    def compose(self, sequences):
        """
        Compose a batch of sequences into a padded tensor.

        Args:
            sequences (List[Tensor]):
                A List of tensors.

        Returns:
            A padded tensor converted to proper device.
        """
        return pad(sequences, self.pad_index).to(self.device)


class ElmoField(Field):
    def __init__(self, name, backbone, lower=False, tokenize=None, fn=None):
        super(ElmoField, self).__init__(name,
                                        lower=lower,
                                        tokenize=tokenize,
                                        fn=fn,
                                        use_vocab=False)
        assert isinstance(backbone, str)
        self.backbone = str.lower(backbone)
        assert self.backbone in {"hit", "allennlp"}
        self.vocab = []

    def build(self):
        return

    def transform(self, sequences):
        return [self.preprocess(seq) for seq in sequences]

    def compose(self, sequences):
        if self.backbone == 'hit':
            return sequences
        elif self.backbone == 'allennlp':
            # it will effect the os.environ['CUDA_VISIBLE_DEVICES'] = device
            from allennlp.modules.elmo import batch_to_ids
            return batch_to_ids(sequences).to(self.device)
        else:
            raise RuntimeError


class FeatureField(Field):
    def __init__(self, *args, **kwargs):
        self.replace_punct = kwargs.pop(
            'replace_punct') if 'replace_punct' in kwargs else False
        self.ud_feature_templates = kwargs.pop(
            'ud_feature_templates'
        ) if 'ud_feature_templates' in kwargs else False
        self.replace_mode = kwargs.pop(
            'replace_mode') if 'replace_mode' in kwargs else None
        self.language_specific_strip = kwargs.pop(
            'language_specific_strip'
        ) if 'language_specific_strip' in kwargs else False
        super().__init__(*args, **kwargs)

    def build(self, dataset, min_freq=1, embed=None):
        """
        构造vocab，并同时构造features
        tokens未被统一小写

        Args:
            dataset:
            min_freq:
            embed:

        Returns:

        """
        if hasattr(self, "vocab"):
            return
        sequences = getattr(dataset, self.name)
        tokens = Counter(token for seq in sequences
                         for token in self.preprocess(seq))
        self.vocab = Vocab(specials=self.specials, unk_index=self.unk_index)

        word_features = Counter()
        unigram_features = Counter()
        bigram_features = Counter()
        trigram_features = Counter()
        for token, freq in tokens.items():
            token = token.lower() if self.lower else token
            word_features.update({token: freq})
            if self.language_specific_strip:
                token = strip_word(token, self.replace_mode)
            if len(token) >= 1:
                unigram_features.update({token[-1:]: freq})
            if len(token) >= 2:
                bigram_features.update({token[-2:]: freq})
            if len(token) >= 3:
                trigram_features.update({token[-3:]: freq})
        self.word_vocab = Vocab(word_features,
                                min_freq,
                                specials=self.specials,
                                unk_index=self.unk_index)
        self.unigram_vocab = Vocab(unigram_features,
                                   min_freq,
                                   specials=self.specials,
                                   unk_index=self.unk_index)
        self.bigram_vocab = Vocab(bigram_features,
                                  min_freq,
                                  specials=self.specials,
                                  unk_index=self.unk_index)
        self.trigram_vocab = Vocab(trigram_features,
                                   min_freq,
                                   specials=self.specials,
                                   unk_index=self.unk_index)

        word_features = [0, self.unk_index]
        unigram_features = [0, self.unk_index]
        bigram_features = [0, self.unk_index]
        trigram_features = [0, self.unk_index]
        morphology_features = [[0, 0, 0], [0, 0, 0]]
        feature_dict = {self.pad: 0, self.unk: 1}
        self.feature_dict = feature_dict
        stoi = {self.pad: 0, self.unk: 1}

        for token in tokens:
            feature = self.get_feature(token)
            if feature not in feature_dict:
                feature_dict[feature] = len(feature_dict)
                word_features.append(feature[0])
                unigram_features.append(feature[1])
                bigram_features.append(feature[2])
                trigram_features.append(feature[3])
                morphology_features.append(list(feature[-3:]))
            stoi[token] = feature_dict[feature]

        self.vocab.stoi = stoi

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.word_features = torch.tensor(word_features).long().to(device)
        self.unigram_features = torch.tensor(unigram_features).long().to(
            device)
        self.bigram_features = torch.tensor(bigram_features).long().to(device)
        self.trigram_features = torch.tensor(trigram_features).long().to(
            device)
        self.morphology_features = torch.tensor(morphology_features).long().to(
            device)

    def get_feature(self, token):
        if self.ud_feature_templates:
            last_feature = 1 if not ispunct(
                replace_punct(token, self.replace_mode)) else 0
        else:
            last_feature = 1 if str.isupper(token[0]) else 0
        token = token.lower() if self.lower else token
        striped_token = token
        if self.language_specific_strip:
            striped_token = strip_word(token, self.replace_mode)
        return (self.word_vocab.stoi[token],
                self.unigram_vocab.stoi[striped_token[-1:]],
                self.bigram_vocab.stoi[striped_token[-2:]],
                self.trigram_vocab.stoi[striped_token[-3:]],
                1 if re.search(r"\d", token) else 0, 1 if "-" in token else 0,
                last_feature)

    def transform(self, sequences):
        """
        Turns a list of sequences that use this field into tensors.

        Each sequence is first preprocessed and then numericalized if needed.

        Args:
            sequences (List[List[str]]):
                A List of sequences.

        Returns:
            sequences (List[Tensor]):
                A list of tensors transformed from the input sequences.
        """

        sequences = [self.preprocess(seq) for seq in sequences]
        if self.use_vocab:
            sequences = [[
                self.vocab[token] if token in self.vocab.stoi else
                self.feature_dict.get(self.get_feature(token), 1)
                for token in seq
            ] for seq in sequences]
        if self.bos:
            sequences = [[self.bos_index] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [self.eos_index] for seq in sequences]
        sequences = [torch.tensor(seq) for seq in sequences]

        return sequences

    @property
    def n_word_features(self):
        return len(self.word_vocab)

    @property
    def n_unigram_features(self):
        return len(self.unigram_vocab)

    @property
    def n_bigram_features(self):
        return len(self.bigram_vocab)

    @property
    def n_trigram_features(self):
        return len(self.trigram_vocab)

    @property
    def n_morphology_features(self):
        return 3

    @property
    def features(self):
        return (self.word_features, self.unigram_features,
                self.bigram_features, self.trigram_features,
                self.morphology_features)

    def preprocess(self, sequence):
        """
        Load a single example using this field, tokenizing if necessary.
        The sequence will be first passed to `self.fn` if available.
        If `self.tokenize` is not None, the input will be tokenized.
        Then the input will be optionally lowercased.

        Args (List):
            The sequence to be preprocessed.

        Returns:
            sequence (List):
                the preprocessed sequence.
        """

        if self.replace_punct:
            sequence = replace_punct_fn(sequence, self.replace_mode)
        sequence = replace_digit_fn(sequence)

        return sequence


class SubwordField(Field):
    """
    A field that conducts tokenization and numericalization over each token rather the sequence.

    This is customized for models requiring character/subword-level inputs, e.g., CharLSTM and BERT.

    Args:
        fix_len (int):
            A fixed length that all subword pieces will be padded to.
            This is used for truncating the subword pieces that exceed the length.
            To save the memory, the final length will be the smaller value
            between the max length of subword pieces in a batch and fix_len.

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        >>> field = SubwordField('bert',
                                 pad=tokenizer.pad_token,
                                 unk=tokenizer.unk_token,
                                 bos=tokenizer.cls_token,
                                 eos=tokenizer.sep_token,
                                 fix_len=20,
                                 tokenize=tokenizer.tokenize)
        >>> field.vocab = tokenizer.get_vocab()  # no need to re-build the vocab
        >>> field.transform([['This', 'field', 'performs', 'token-level', 'tokenization']])[0]
        tensor([[  101,     0,     0],
                [ 1188,     0,     0],
                [ 1768,     0,     0],
                [10383,     0,     0],
                [22559,   118,  1634],
                [22559,  2734,     0],
                [  102,     0,     0]])
    """
    def __init__(self, *args, **kwargs):
        self.fix_len = kwargs.pop('fix_len') if 'fix_len' in kwargs else 0
        super().__init__(*args, **kwargs)

    def build(self, dataset, min_freq=1, embed=None):
        if hasattr(self, 'vocab'):
            return
        sequences = getattr(dataset, self.name)
        counter = Counter(piece for seq in sequences for token in seq
                          for piece in self.preprocess(token))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab[tokens]] = embed.vectors

    def transform(self, sequences):
        sequences = [[self.preprocess(token) for token in seq]
                     for seq in sequences]
        if self.fix_len <= 0:
            self.fix_len = max(
                len(token) for seq in sequences for token in seq)
        if self.use_vocab:
            sequences = [[[self.vocab[i]
                           for i in token] if token else [self.unk]
                          for token in seq] for seq in sequences]
        if self.bos:
            sequences = [[[self.bos_index]] + seq for seq in sequences]
        if self.eos:
            sequences = [seq + [[self.eos_index]] for seq in sequences]
        lens = [
            min(self.fix_len, max(len(ids) for ids in seq))
            for seq in sequences
        ]
        sequences = [
            pad([torch.tensor(ids[:i]) for ids in seq], self.pad_index, i)
            for i, seq in zip(lens, sequences)
        ]

        return sequences
