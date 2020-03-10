import torch
import torch.nn as nn
from utils import to_input_tensor

# this fn from: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class AverageLinear(nn.Module):
    def __init__(self, num_classes, pretrained_embeddings, embeddings_df):
        super(AverageLinear, self).__init__()
        self.embeddings_df = embeddings_df # need this to look up glove vector indices

        self.num_classes = num_classes
        self.embedding, self.num_embeddings, self.embed_dim = create_emb_layer(pretrained_embeddings, non_trainable=True)
        self.linear = nn.Linear(2*self.embed_dim, self.num_classes)

    def forward(self, context, word):
        # convert word lists to indices
        context_input = to_input_tensor(context, self.embeddings_df, is_contexts=True)
        word_input = to_input_tensor(word, self.embeddings_df, is_contexts=False)

        # get embeddings of all words in context and the word
        context_embed = self.embedding(context_input)
        word_embed = self.embedding(word_input)

        # take average of context words to combine
        # context_embed = (b, con_len, embed_dim)
        context_embed = torch.mean(context_embed, dim=1)

        # run through linear layer
        concat_features = torch.cat((word_embed, context_embed), axis=1)
        linear_output = self.linear(concat_features)
        return linear_output
