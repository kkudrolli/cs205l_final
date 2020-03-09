from dataset import SemCor

data = SemCor()

for d in data.data:
    print(d)

# TODO figure out way to store and upload embeddings into model
# TODO write train/test loop, backprop code
# TODO add models: can try averaging word vectors first + linear layer, RNN encoder that takes in the context (so it's a fixed length)
# TODO train glove more and add code to load them into model
# TODO compute the max number of senses in the dataset and just ignore or mask off
# TODO related look up word senses for each word?
