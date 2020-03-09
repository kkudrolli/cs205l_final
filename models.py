import torch

class AverageLinear(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.linear = nn.Linear(..., self.num_classes)

        # TODO add embeddings

    def forward(self, ...):
        # TODO get embeddings of all words in sentence (context?)
        # TODO take average
        # TODO run through linear layer and softmax
        return 1
