from nltk.corpus import semcor


class Example:
    def __init__(self, sent, word, sense, pos):
        self.sent = sent # input
        self.word = word # input
        self.sense = sense # label
        self.pos = pos
    
    def __str__(self):
        return "({}.{}.{:02d}) : {}".format(self.word, self.pos, self.sense, self.sent)

class SemCor:
    def __init__(self):
        self.data = []

        self.parse()


    def parse_sense(self, sense_str):
        sense_list = sense_str.split(';') 
        # in some cases multiple given ;-separate, just take first
        return int(sense_list[0])

    def parse(self):
        tagged_sents = semcor.tagged_sents(tag='sense')
        sents = semcor.sents()

        # tagged_sents returns senses of each word/group of words
        for sent, tag in zip(sents, tagged_sents):
            # TODO need to lemmatize word in sent?
            #print(sent)

            for entry in tag:
                #print(entry)
                
                # check for no sense tag or multiword entries
                # TODO is it ok to disclude multiword entries?
                if entry.label() and len(entry.leaves()) == 1: 
                    entry = entry.label().split('.')

                    if len(entry) == 3: # check for (word.pos.nn) entry
                        word, pos, sense = entry

                        new_ex = Example(sent, word, self.parse_sense(sense), pos)
                        # add to data set
                        self.data.append(new_ex)
                        # TODO for now just take first sense found in sentence
                        break
                
