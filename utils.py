# some misc functions
import numpy as np
import torch
import math

# return the train, val, and test splits
def split_data(dataset, train_split=0.8, val_split=0.1, test_split=0.1):
    assert(train_split + val_split + test_split == 1.0)

    N = len(dataset.data)
    train_data = dataset.data[:int(train_split*N)]
    val_data = dataset.data[int(train_split*N) : int((train_split+val_split)*N)]
    test_data = dataset.data[int((train_split+val_split)*N):]

    return train_data, val_data, test_data
    

def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        contexts = [e.context for e in examples]
        words = [e.word for e in examples]
        senses = [e.sense for e in examples]

        yield contexts, words, torch.LongTensor(senses)


def words2indices(words, is_contexts, embeddings_df):
    indices = []
    if is_contexts:
        contexts = words
        context_indices = []
        for context in contexts:
            for word in context:
                # look up word id from glove vector
                # TODO fix this unk handling
                if word in embeddings_df:
                    index = embeddings_df.index.get_loc(word)
                else:
                    index = 0
                context_indices.append(index)
            indices.append(context_indices)
    else:
        for word in words:
            if word in embeddings_df:
                index = embeddings_df.index.get_loc(word)
            else:
                index = 0
            indices.append(index)

    return indices
    

def to_input_tensor(words, embeddings_df, is_contexts=False):
    word_ids = words2indices(words, is_contexts, embeddings_df)
    return torch.LongTensor(word_ids)
