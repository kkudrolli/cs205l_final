from nltk.corpus import semcor
from nltk.corpus import wordnet as wn

class Example:
    def __init__(self, context, word, sense, pos, num_senses):
        self.context = context # input, words surrounding word to classify
        self.word = word # input, word to classify
        self.sense = sense # label
        self.pos = pos # part-of-speech, unused for now
        self.num_senses = num_senses
    
    def __str__(self):
        return "({}.{}.{:02d}) : {}".format(self.word, self.pos, self.sense, self.sent)

class SemCor:
    def __init__(self, context_size):
        self.data = []
        self.max_num_senses = 0
        self.context_size = context_size # how many words on left and right to take

        self.parse()

    def get_context(self, sent, word_idx):
        # TODO improve the padding
        context = sent[max(word_idx-self.context_size,0) : word_idx-self.context_size+1]
        desired_len = 2*self.context_size+1
        if len(context) < desired_len:
            context += [''] * (desired_len - len(context))
        return context

    def count_senses(self, word):
        num_senses = len(wn.synsets(word))
        self.max_num_senses = max(self.max_num_senses, num_senses)
        return num_senses

    def parse_sense(self, sense_str):
        sense_list = sense_str.split(';') 
        # in some cases multiple given ;-separate, just take first
        return max(int(sense_list[0])-1, 0) # TODO is there ever a 0th sense?

    def parse(self):
        tagged_sents = semcor.tagged_sents(tag='sense')
        sents = semcor.sents()

        # tagged_sents returns senses of each word/group of words
        for sent, tag in zip(sents, tagged_sents):

            word_idx = 0
            for entry in tag:
                # check for no sense tag or multiword entries
                # TODO is it ok to exclude multiword entries?
                entry_len = len(entry.leaves())
                if entry.label() and entry_len == 1 and type(entry.label()) != str: 
                    #import pdb; pdb.set_trace()
                    entry = entry.label().synset().name().split('.')

                    if len(entry) == 3: # check for (word.pos.nn) entry
                        word, pos, sense = entry
                        
                        num_senses = self.count_senses(word)
                        context = self.get_context(sent, word_idx)

                        new_ex = Example(context, word, self.parse_sense(sense), pos, num_senses)
                        # add to data set
                        self.data.append(new_ex)
                        # TODO for now just take first sense found in sentence
                        break
                word_idx += entry_len # one entry might be multiple words
