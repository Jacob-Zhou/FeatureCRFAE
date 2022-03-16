# Feature_CRF_AE

`Feature_CRF_AE` provides a implementation of [Bridging Pre-trained Language Models and Hand-crafted Features for Unsupervised POS Tagging](hlt.suda.edu.cn/LA/papers/acl-findings-hqzhou-bridging.pdf):

```bib
@inproceedings{zhou-etal-2022-Bridging,
  title     = {Bridging Pre-trained Language Models and Hand-crafted Features for Unsupervised POS Tagging},
  author    = {Zhou, houquan and Li, yang and Li, Zhenghua and Zhang Min},
  booktitle = {Findings of ACL},
  year      = {2022},
  url       = {?},
  pages     = {?--?}
}
```

Please concact `Jacob_Zhou \at outlook.com` if you have any questions.

## Contents

* [Contents](#contents)
* [Installation](#installation)
* [Performance](#performance)
* [Usage](#usage)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Predict](#predict)

## Installation

`Feature_CRF_AE` can be installing from source:
```sh
$ git clone https://github.com/Jacob-Zhou/FeatureCRFAE && cd FeatureCRFAE
$ bash scripts/setup.sh
```

The following requirements will be installed in `scripts/setup.sh`:
* `python`: 3.7
* [`allennlp`](https://github.com/allenai/allennlp): 1.2.2
* [`pytorch`](https://github.com/pytorch/pytorch): 1.6.0
* [`transformers`](https://github.com/huggingface/transformers): 3.5.1
* `h5py`: 3.1.0
* `matplotlib`: 3.3.1
* `nltk`: 3.5
* `numpy`: 1.19.1
* `overrides`: 3.1.0
* `scikit_learn`: 1.0.2
* `seaborn`: 0.11.0
* `tqdm`: 4.49.0

For WSJ data, we use the ELMo representations of `elmo_2x4096_512_2048cnn_2xhighway_5.5B` from [**AllenNLP**](https://allenai.org/allennlp/software/elmo).
For UD data, we use the ELMo representations released by [**HIT-SCIR**](https://github.com/HIT-SCIR/ELMoForManyLangs).

The corresponding data and ELMo models can be download as follows:
```sh
# 1) UD data and ELMo models:
$ bash scripts/prepare_data.sh
# 2) UD data, ELMo models as well as WSJ data 
#    [please replace ~/treebank3/parsed/mrg/wsj/ with your path to LDC99T42]
$ bash scripts/prepare_data.sh ~/treebank3/parsed/mrg/wsj/
```

## Performance

### WSJ-All

|  Seed  |  M-1  |  1-1  |  VM   |
| :----: | :---: | :---: | :---: |
| 0      | 84.29 | 70.03 | 78.43 |
| 1      | 82.34 | 64.42 | 77.27 |
| 2      | 84.68 | 62.78 | 77.83 |
| 3      | 82.55 | 65.00 | 77.35 |
| 4      | 82.20 | 66.69 | 77.33 |
| Avg.   | 83.21 | 65.78 | 77.64 |
| Std.   | 1.18  | 2.75  | 0.49  |

### WSJ-Test

|  Seed  |  M-1  |  1-1  |  VM   |
| :----: | :---: | :---: | :---: |
| 0      | 81.99 | 64.84 | 76.86 |
| 1      | 82.52 | 61.46 | 76.13 |
| 2      | 82.33 | 61.15 | 75.13 |
| 3      | 78.11 | 58.80 | 72.94 |
| 4      | 82.05 | 61.68 | 76.21 |
| Avg.   | 81.40 | 61.59 | 75.45 |
| Std.   | 1.85  | 2.15  | 1.54  |

## Usage

We give some examples on `scripts/examples.sh`.
Before run the code you should activate the virtual environment by:
```sh
$ . scripts/set_environment.sh
```

### Training

To train a model from scratch, it is preferred to use the command-line option, which is more flexible and customizable.
Here are some training examples:
```sh
$ python -u -m tagger.cmds.crf_ae train \
    --conf configs/crf_ae.ini \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --train data/wsj/total.conll \
    --evaluate data/wsj/total.conll \
    --path save/crf_ae_wsj
```

```sh
$ python -u -m tagger.cmds.crf_ae train \
    --conf configs/crf_ae.ini \
    --ud-mode \
    --ud-feature \
    --ignore-capitalized \
    --language-specific-strip \
    --feat-min-freq 14 \
    --language de \
    --encoder elmo \
    --plm elmo_models/de \
    --train data/ud/de/total.conll \
    --evaluate data/ud/de/total.conll \
    --path save/crf_ae_de
```

For more instructions on training, please type `python -m tagger.cmds.[crf_ae|feature_hmm] train -h`.

Alternatively, We provides some equivalent command entry points registered in `setup.py`:
`crf-ae` and `feature-hmm`.
```sh
$ crf-ae train \
    --conf configs/crf_ae.ini \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --train data/wsj/total.conll \
    --evaluate data/wsj/total.conll \
    --path save/crf_ae
```

### Evaluation

```sh
$ python -u -m tagger.cmds.crf_ae evaluate \
    --conf configs/crf_ae.ini \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --data data/wsj/total.conll \
    --path save/crf_ae
```

### Predict

```sh
$ python -u -m tagger.cmds.crf_ae predict \
    --conf configs/crf_ae.ini \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --data data/wsj/total.conll \
    --path save/crf_ae \
    --pred save/crf_ae/pred.conll
```