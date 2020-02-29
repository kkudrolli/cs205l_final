
# Inspired by: https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html

import sys
from gensim.corpora import WikiCorpus

def make_corpus(in_f, out_f):

    with open(out_f, 'w') as output:
        wiki = WikiCorpus(in_f)
        print("Created corpus object")

        for i,text in enumerate(wiki.get_texts()):
            output.write(bytes(" ".join(text), 'utf-8').decode('utf-8') + '\n')
            if i % 10 == 0:
                print("Processed {} articles".format(i))

        print("Finished creating corpus")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Input error")
        sys.exit(1)
    wiki_file = sys.argv[1]
    make_corpus(wiki_file, 'wiki_en.txt')
