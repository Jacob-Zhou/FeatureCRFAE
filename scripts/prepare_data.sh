#!/bin/bash
set -o nounset
set -o pipefail

if [ $# -lt 1 ];then
    echo "Warning: WSJ will not be download. If you want WSJ, you should input the path to WSJ as follows:" >&2
    echo "  bash scripts/prepare_data.sh ~/treebank3/parsed/mrg/wsj/" >&2
    have_wsj=0
else
    have_wsj=1
    wsj_path=$1
    if [ ! "${wsj_path:0:1}" = "/" ];then
        wsj_path=`pwd`/$wsj_path
    fi

    lowercase_mrgs=$(ls $wsj_path/*/*.mrg 2> /dev/null | wc -l);
    uppercase_mrgs=$(ls $wsj_path/*/*.MRG 2> /dev/null | wc -l);

    is_mrgs_uppercase=0

    set -o errexit

    if [ $lowercase_mrgs -ne 0 ];then
        is_mrgs_uppercase=0
    elif [ $uppercase_mrgs -ne 0 ];then
        is_mrgs_uppercase=1
    else
        echo "WSJ PATH may wrong!"
        exit 1
    fi
fi

if [ ! -d data ];then
    mkdir data
fi

if [ ! -d data/ud ];then
    mkdir data/ud
    wget -O data/ud/ud_data.tar.gz https://github.com/ryanmcd/uni-dep-tb/raw/master/universal_treebanks_v2.1.tar.gz
    tar -zxf data/ud/ud_data.tar.gz -C data/ud
    mv data/ud/universal_treebanks_v2.1/std/* data/ud
    rm -r data/ud/universal_treebanks_v2.1
    rm data/ud/ud_data.tar.gz
    if [ $have_wsj -ne 0 ]; then
        echo "Generating UD-english and WSJ"
        # the english data need to be generated from LDC99T42
        # correct the typo on data/ud/en/generate-data.sh
        sed -i -e 's/univiersal/universal/g' data/ud/en/generate-data.sh
        sed -i -e "s#treebank3/parsed/mrg/wsj/#$wsj_path#g" data/ud/en/generate-data.sh
        # don't remove the tmp.conll yet
        sed -i 's/rm -f $split-tmp.mrg $split-stanford-raw.conll/rm -f $split-tmp.mrg/g' data/ud/en/generate-data.sh
        if [ $is_mrgs_uppercase -eq 1 ];then
            sed -i -e 's/mrg/MRG/g' data/ud/en/generate-data.sh
        fi
        cur_path=`pwd`
        cd data/ud/en/
        bash generate-data.sh
        if [ ! -d $cur_path/data/wsj ];then
            mkdir $cur_path/data/wsj
            splits=(test dev train)
            for split in ${splits[@]};do
                cp $split-stanford-raw.conll $cur_path/data/wsj/$split.conll
            done
            # build WSJ-All data
            echo "create Stanford dependencies for WSJ-All ..."
            if [ $is_mrgs_uppercase -eq 1 ];then
                cat $wsj_path/*/*.MRG > $cur_path/data/wsj/total-tmp.MRG
            else
                cat $wsj_path/*/*.mrg > $cur_path/data/wsj/total-tmp.MRG
            fi
            java -cp stanford-parser-v1.6.8.jar \
                -Xmx2g \
                edu.stanford.nlp.trees.EnglishGrammaticalStructure \
                -treeFile $cur_path/data/wsj/total-tmp.MRG \
                -conllx -basic -makeCopulaHead -keepPunct > $cur_path/data/wsj/total.conll
            rm -f $cur_path/data/wsj/total-tmp.MRG
        fi
        rm *-stanford-raw.conll
        cd $cur_path
    fi
    # merge train/dev/test into a whole file
    for path in data/ud/*;do
        if test -d $path && ([ $path != 'data/ud/en' ] || [ $have_wsj -ne 0 ]);then
            cat $path/*-universal-*.conll > $path/total.conll
        fi
    done
elif [ ! -d data/wsj ] && [ $have_wsj -ne 0 ] ;then
    echo "Generating WSJ"
    mkdir data/wsj
    if [ $is_mrgs_uppercase -eq 1 ];then
        mrg_suffix=MRG
    else
        mrg_suffix=mrg
    fi
    cat $wsj_path/0[2-9]/*.$mrg_suffix $wsj_path/1*/*.$mrg_suffix \
    $wsj_path/2[0-1]/*.$mrg_suffix > data/wsj/train-tmp.MRG
    cat $wsj_path/22/*.$mrg_suffix > data/wsj/dev-tmp.MRG
    cat $wsj_path/23/*.$mrg_suffix > data/wsj/test-tmp.MRG
    cat $wsj_path/*/*.$mrg_suffix > data/wsj/total-tmp.MRG
    default_stanford_parser=data/ud/en/stanford-parser-v1.6.8.jar
    if [ ! -f $default_stanford_parser ];then
        mkdir data/parser
        wget -O data/parser/stanford_parser.tgz https://nlp.stanford.edu/software/stanford-parser-2011-08-04.tgz
        tar -xf data/parser/stanford_parser.tgz -C data/parser
        default_stanford_parser=data/parser/stanford-parser-2011-08-04/stanford-parser.jar
    fi
    # Create stanford dependencies
    splits=(test dev train)
    for split in ${splits[@]}; do
        echo "create Stanford dependencies for split WSJ-"$split" ..."
        java -cp $default_stanford_parser \
            -Xmx2g \
            edu.stanford.nlp.trees.EnglishGrammaticalStructure \
            -treeFile data/wsj/$split-tmp.MRG \
            -conllx -basic -makeCopulaHead -keepPunct > data/wsj/$split.conll
        rm -f data/wsj/$split-tmp.MRG
    done
    echo "create Stanford dependencies for WSJ-All ..."
    java -cp $default_stanford_parser \
        -Xmx2g \
        edu.stanford.nlp.trees.EnglishGrammaticalStructure \
        -treeFile data/wsj/total-tmp.MRG \
        -conllx -basic -makeCopulaHead -keepPunct > data/wsj/total.conll
    rm -f data/wsj/total-tmp.MRG
fi

# download ELMo models
function download() {
    # para: $1(langs name) $2(url)
    model_path=elmo_models/$1
    if [ ! -d $model_path ];then
        mkdir $model_path
        wget -O $model_path/model.zip $2
        unzip $model_path/model.zip -d $model_path
        rm $model_path/model.zip
        python scripts/setup_model_config.py $model_path `pwd`
    fi
}

if [ ! -d elmo_models ];then
    mkdir elmo_models
fi
# download models from https://github.com/HIT-SCIR/ELMoForManyLangs
# German
download de http://vectors.nlpl.eu/repository/11/142.zip
# English
download en http://vectors.nlpl.eu/repository/11/144.zip
# Spanish
download es http://vectors.nlpl.eu/repository/11/145.zip
# French
download fr http://vectors.nlpl.eu/repository/11/150.zip
# Indonesian
download id http://vectors.nlpl.eu/repository/11/158.zip
# Italian
download it http://vectors.nlpl.eu/repository/11/159.zip
# Japanese
download ja http://vectors.nlpl.eu/repository/11/160.zip
# Korean
download ko http://vectors.nlpl.eu/repository/11/161.zip
# Portuguese
download pt http://vectors.nlpl.eu/repository/11/168.zip
# Swedish
download sv http://vectors.nlpl.eu/repository/11/173.zip

# download models from AllenNLP
if [ ! -d elmo_models/allennlp ];then
    mkdir elmo_models/allennlp
    mkdir elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B
    wget -O elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B/weights.hdf5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
    wget -O elmo_models/allennlp/elmo_2x4096_512_2048cnn_2xhighway_5.5B/options.json https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json
fi
