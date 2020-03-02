import argparse
import numpy as np


def read_corpus(corpus_file, num_lines=-1):
    corpus = []
    with open(corpus_file, 'r') as f:
        for i,doc in enumerate(f.readlines()):
            doc_words_list = doc.split()
            corpus.append(doc_words_list)

            # if specified,  stop after certain number of line
            if num_lines > 0 and num_lines == i: 
                return corpus
    return corpus
    

def distinct_words(corpus, vocab_file=None):
    if vocab_file:
        with open(args.vocab_file, 'r') as f:
            words = [x.rstrip().split(' ')[0] for x in f.readlines()]
            num_words = len(words)
            return words, num_words
    else:
        words = []
        num_words = -1
        
        word_set = set([])
        for doc in corpus:
            for word in doc:
                word_set.add(word)

        word_list = list(word_set)
        words = sorted(word_list)
        num_words = len(word_list)

        return words, num_words


def create_cooccur_matrix(corpus, vocab_file=None, window_size=15):
    words, num_words = distinct_words(corpus, vocab_file)
    
    M = np.zeros((num_words, num_words))
    word2Ind = {}

    # populate dict with indices
    for i, word in enumerate(words):
        word2Ind[word] = i

    for doc in corpus:
        for i,word in enumerate(doc):
            # index of center of word
            row = word2Ind[word]

            # create list of words in window
            window = doc[i-window_size:i] + doc[i+1:i+window_size+1]

            # for each word in window increment matrix cell for (center, window)
            for win_word in window:
                col = word2Ind[win_word]
                M[row][col] += 1

    return M

def write_matrix_to_file(cooccur_matrix, cooccur_write_file):
    np.savetxt(cooccur_write_file, cooccur_matrix, delimeter=',')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--corpus_file', default='wiki_en.txt', type=str)
    parser.add_argument('--cooccur_write_file', default='cooccur.txt', type=str)
    parser.add_argument('--num_lines', default=-1, type=int)
    args = parser.parse_args()

    corpus = read_corpus(args.corpus_file, args.num_lines)
    cooccur_matrix = create_cooccur_matrix(corpus, args.vocab_file)
    write_matrix_to_file(cooccur_matrix, args.cooccur_write_file)

if __name__ == "__main__":
    main()
