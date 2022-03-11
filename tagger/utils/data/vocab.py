from collections import defaultdict
from collections.abc import Iterable


class Vocab(object):
    """
    Defines a vocabulary object that will be used to numericalize a field.

    Args:
        counter (Counter):
            Counter object holding the frequencies of each value found in the data.
        min_freq (int):
            The minimum frequency needed to include a token in the vocabulary.
        specials (List[str]):
            The list of special tokens (e.g., pad, unk, bos and eos) that will be prepended to the vocabulary.
        unk_index (int):
            The index of unk token.
    """
    def __init__(self, counter=None, min_freq=1, specials=None, unk_index=0):
        self.itos = list(specials) if specials is not None else []
        self.stoi = defaultdict(lambda: unk_index)
        self.stoi.update({token: i for i, token in enumerate(self.itos)})
        if counter is not None:
            self.extend(
                [token for token, freq in counter.items() if freq >= min_freq])
        self.unk_index = unk_index
        self.n_init = len(self.itos)

    def __len__(self):
        return self.n_init

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.stoi[key]
        elif not isinstance(key, Iterable):
            return self.itos[key]
        elif isinstance(key[0], str):
            return [self.stoi.get(i, self.unk_index) for i in key]
        else:
            return [self.itos[i] for i in key]

    def __contains__(self, token):
        return token in self.stoi

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        stoi = defaultdict(lambda: self.unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    def extend(self, tokens):
        self.itos.extend(sorted(set(tokens).difference(self.stoi)))
        self.stoi.update({token: i for i, token in enumerate(self.itos)})
