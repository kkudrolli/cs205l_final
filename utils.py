
# return the train, val, and test splits
def split_data(dataset, train_split=0.8, val_split=0.1, test_split=0.1):
    assert(train_split + val_split + test_split == 1.0)

    N = len(dataset.data)
    train_data = dataset.data[:int(train_split)*N]
    val_data = dataset.data[int(train_split) : int((train_split+val_split)*N)]
    test_data = dataset.data[int((train_split+val_split)*N):]

    return train_data, val_data, test_data
    
