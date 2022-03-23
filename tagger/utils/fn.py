import re
import unicodedata
from functools import lru_cache
from tagger.utils.common import language_specific_punct_replace, punct_replace

import matplotlib.pyplot as plt
import seaborn as sns


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)

def strip_word(token, replace_mode):
    if replace_mode == "it":
        return token[:-1]
    elif replace_mode in {"es", "pt", "pt-br", "fr"}:
        if token[-1] == "s":
            return token[:-2]
        else:
            return token[:-1]
    elif replace_mode == "de":
        return token[:-2]
    elif replace_mode == "en" and token[-1] == "s":
        return token[:-1]
    else:
        return token

def replace_punct(token, replace_mode):
    if token in punct_replace:
        token = punct_replace[token]
    if replace_mode in language_specific_punct_replace:
        if token in language_specific_punct_replace[replace_mode]:
            token = language_specific_punct_replace[replace_mode][token]
    return token

def strip_punct(token):
    return "".join([char for char in token if not unicodedata.category(char).startswith('P')])

def replace_punct_fn(sequence, replace_mode):
    sequence = [replace_punct(token, replace_mode) for token in sequence]
    return ["...." if ispunct(token) else strip_punct(token) for token in sequence]

def pad(tensors, padding_value=0, total_length=None, padding_side='right'):
    size = [len(tensors)] + [
        max(tensor.size(i) for tensor in tensors)
        for i in range(len(tensors[0].size()))
    ]
    if total_length is not None:
        assert total_length >= size[1]
        size[1] = total_length
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[
            slice(-i, None) if padding_side == 'left' else slice(0, i)
            for i in tensor.size()
        ]] = tensor
    return out_tensor

def get_masks(sequence):
    return [1] * len(sequence)

@lru_cache(maxsize=1024)
def replace_digit(token):
    return re.sub(r"[0-9]+", '0', token)

def replace_digit_fn(sequence):
    return [replace_digit(token) for token in sequence]

def preprocess_hmm_fn(sequence, language=None, language_specific_strip=False, replace_punct=False):
    new_seq = []
    replace_mode = language if language_specific_strip else None
    for token in sequence:
        if replace_punct:
            token = replace_punct(token, replace_mode)
        if language_specific_strip:
            token = strip_word(token, replace_mode)
        token = replace_digit(token)
        new_seq.append(token)
    return new_seq

def none_or_str(value):
    if value in ['None', 'NONE', 'none', 'null']:
        return None
    return value

def heatmap(corr, labels=None, name='matrix', match=None):
    sns.set(style="white")

    shape = corr.t().shape
    assert len(shape) == 2
    shape = (shape[0], shape[1])

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=shape)

    cmap = "RdBu"

    ylabels = [f"{i :2d}" for i in range(shape[0])]

    if match is not None:
        match_ids = [0] * shape[0]
        for p_idx, g_idx in enumerate(match):
            match_ids[g_idx] = p_idx
        corr = corr[match_ids]
        ylabels = [ylabels[idx] for idx in match_ids]

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(100. * corr / corr.sum().float(),
                cmap=cmap,
                center=0,
                ax=ax,
                square=True,
                linewidths=.5,
                vmax=8.,
                annot=True,
                xticklabels=False if labels is None else labels,
                yticklabels=ylabels,
                cbar=False)
    plt.margins(0, 0)
    plt.subplots_adjust(left=0.04, bottom=0., right=0.96, top=1.)
    plt.savefig(f'{name}.png')
    plt.close()
