#!/bin/bash

EXT="/media/kkudrolli/Expansion Drive"
CMD="$1"
PYTHON=python3

if [ "$CMD" = "cooccur" ]; then
    $PYTHON cooccur.py \
        --vocab_file "$EXT/enwiki/vocab.txt" \
        --corpus_file "$EXT/enwiki/wiki_en.txt" \
        --cooccur_write_file "$EXT/enwiki/cooccur.csv" \
        --num_lines 5 
fi

if [ "$CMD" = "wiki" ]; then
    $PYTHON create_wiki_corpus.py $EXT/enwiki-latest-pages-articles.xml.bz2
fi
