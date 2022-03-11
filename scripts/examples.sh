# WSJ-All
## feature_hmm training
python -u -m tagger.cmds.feature_hmm train \
    --seed 0 \
    --device 2 \
    --conf configs/feature_hmm.ini \
    --train data/wsj/train.conllx \
    --evaluate data/wsj/dev.conllx \
    --test data/wsj/test.conllx \
    --epochs 50 \
    --path save/feature_hmm_wsf

## feature_hmm evaluating
python -u -m tagger.cmds.feature_hmm evaluate \
    --seed 0 \
    --device 3 \
    --conf configs/feature_hmm.ini \
    --data data/wsj/test.conllx \
    --path save/feature_hmm_wsf

## feature_hmm predicting
python -u -m tagger.cmds.feature_hmm predict \
    --seed 0 \
    --device 3 \
    --conf configs/feature_hmm.ini \
    --data data/wsj/test.conllx \
    --path save/feature_hmm_wsf \
    --pred save/feature_hmm_wsf/pred.conll

## crf_ae training
python -u -m tagger.cmds.crf_ae train \
    --seed 4 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --train data/wsj/total.conll \
    --evaluate data/wsj/total.conll \
    --path save/elmo_wa_4

## crf_ae evaluating
python -u -m tagger.cmds.crf_ae evaluate \
    --seed 0 \
    --device 3 \
    --conf configs/crf_ae.ini \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --data data/wsj/total.conll \
    --path save/elmo_wa

## crf_ae predicting
python -u -m tagger.cmds.crf_ae predict \
    --seed 0 \
    --device 3 \
    --conf configs/crf_ae.ini \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --data data/wsj/total.conll \
    --path save/elmo_wa \
    --pred save/elmo_wa/pred.conll

# WSJ-Split Full
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --train data/wsj/train.conllx \
    --evaluate data/wsj/dev.conllx \
    --test data/wsj/test.conllx \
    --path save/elmo_wsf

# WSJ-Split - Feature
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --without-feature \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --train data/wsj/train.conllx \
    --evaluate data/wsj/dev.conllx \
    --test data/wsj/test.conllx \
    --path save/elmo_wsmf

# WSJ-Split LSTM
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae_lstm.ini \
    --encoder lstm \
    --n-layers 3 \
    --train data/wsj/train.conllx \
    --evaluate data/wsj/dev.conllx \
    --test data/wsj/test.conllx \
    --path save/elmo_wsl

# WSJ-Split - minus
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --without-fd-repr \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --train data/wsj/train.conllx \
    --evaluate data/wsj/dev.conllx \
    --test data/wsj/test.conllx \
    --path save/elmo_wsmm

# WSJ-Split rand init
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --rand-init \
    --encoder elmo \
    --plm elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B \
    --train data/wsj/train.conllx \
    --evaluate data/wsj/dev.conllx \
    --test data/wsj/test.conllx \
    --epochs 1 \
    --path save/elmo_wsr

# WSJ-Split BERT
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae_bert.ini \
    --encoder bert \
    --plm bert-base-cased \
    --train data/wsj/train.conllx \
    --evaluate data/wsj/dev.conllx \
    --test data/wsj/test.conllx \
    --path save/elmo_wsb

# UD-All de
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
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
    --path save/elmo_dea

# UD-Split Full
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --ud-mode \
    --ud-feature \
    --ignore-capitalized \
    --language-specific-strip \
    --feat-min-freq 14 \
    --language de \
    --encoder elmo \
    --plm elmo_models/de \
    --train data/ud/de/de-universal-train.conll \
    --evaluate data/ud/de/de-universal-dev.conll \
    --test data/ud/de/de-universal-test.conll \
    --path save/elmo_desf

# UD-Split - Feature
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --without-feature \
    --ud-mode \
    --replace-punct \
    --ignore-capitalized \
    --language-specific-strip \
    --feat-min-freq 14 \
    --language de \
    --encoder elmo \
    --plm elmo_models/de \
    --train data/ud/de/de-universal-train.conll \
    --evaluate data/ud/de/de-universal-dev.conll \
    --test data/ud/de/de-universal-test.conll \
    --path save/elmo_desmf

# UD-Split - UD Adjust
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --ud-mode \
    --language-specific-strip \
    --feat-min-freq 14 \
    --language de \
    --encoder elmo \
    --plm elmo_models/de \
    --train data/ud/de/de-universal-train.conll \
    --evaluate data/ud/de/de-universal-dev.conll \
    --test data/ud/de/de-universal-test.conll \
    --path save/elmo_desmuda

# UD-Split - Lg Adjust
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --ud-mode \
    --ud-feature \
    --ignore-capitalized \
    --feat-min-freq 14 \
    --language de \
    --encoder elmo \
    --plm elmo_models/de \
    --train data/ud/de/de-universal-train.conll \
    --evaluate data/ud/de/de-universal-dev.conll \
    --test data/ud/de/de-universal-test.conll \
    --path save/elmo_desmlga

# UD-Split - LSTM
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae_lstm.ini \
    --encoder lstm \
    --n-layers 3 \
    --ud-mode \
    --ud-feature \
    --ignore-capitalized \
    --language-specific-strip \
    --feat-min-freq 14 \
    --language de \
    --train data/ud/de/de-universal-train.conll \
    --evaluate data/ud/de/de-universal-dev.conll \
    --test data/ud/de/de-universal-test.conll \
    --path save/elmo_desl

# UD-Split rand init
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae.ini \
    --rand-init \
    --ud-mode \
    --ud-feature \
    --ignore-capitalized \
    --language-specific-strip \
    --feat-min-freq 14 \
    --language de \
    --encoder elmo \
    --plm elmo_models/de \
    --train data/ud/de/de-universal-train.conll \
    --evaluate data/ud/de/de-universal-dev.conll \
    --test data/ud/de/de-universal-test.conll \
    --path save/elmo_desr

# UD-Split mBERT
python -u -m tagger.cmds.crf_ae train \
    --seed 0 \
    --device 2 \
    --conf configs/crf_ae_bert.ini \
    --encoder bert \
    --plm bert-base-multilingual-cased \
    --seed 0 \
    --ud-mode \
    --ud-feature \
    --ignore-capitalized \
    --language-specific-strip \
    --feat-min-freq 14 \
    --language de \
    --train data/ud/de/de-universal-train.conll \
    --evaluate data/ud/de/de-universal-dev.conll \
    --test data/ud/de/de-universal-test.conll \
    --path save/elmo_desb
