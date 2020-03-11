#!/bin/bash

EXT="/media/kkudrolli/Expansion Drive"
CMD="$1"
PYTHON=python3

if [ "$CMD" = "cooccur_wiki" ]; then
    $PYTHON cooccur.py \
        --vocab_file "$EXT/enwiki/vocab.txt" \
        --corpus_file "$EXT/enwiki/wiki_en.txt" \
        --corpus_name "wiki" \
        --num_lines 5 
fi

if [ "$CMD" = "cooccur_semcor" ]; then
    $PYTHON cooccur.py \
        --corpus_name "semcor" \
        --is_semcor True
fi

if [ "$CMD" = "wiki" ]; then
    $PYTHON create_wiki_corpus.py $EXT/enwiki-latest-pages-articles.xml.bz2
fi

if [ "$CMD" = "train" ]; then
    $PYTHON train.py \
        --emb_file data/giga5_glv.pkl \
        --model_name LSTMEncoder \
        --context_size 1 \
        --lr 0.1 
fi
