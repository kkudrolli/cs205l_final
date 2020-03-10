from dataset import SemCor
from models import AverageLinear
from utils import split_data, batch_iter

import torch
import os
import pickle
import torch.nn as nn
import pandas as pd
import argparse

# TODO add model: RNN encoder that takes in the context (so it's a fixed length)
# TODO look up word senses for each word?
# TODO consider creating co-occurrence matrix with semcor

def evaluate_accuracy(model, data, batch_size):
    num_correct = 0
    for contexts, words, senses in batch_iter(data, batch_size, shuffle=False):
        scores = model(contexts, words)
        preds = torch.argmax(scores)
        num_correct += preds.eq(senses).sum()

    accuracy = (num_correct / len(data))*100
    return accuracy

def validate(model, val_data, epoch, train_iter, batch_size=128):
    model.eval()
    
    with torch.no_grad():
        accuracy = evaluate_accuracy(model, val_data, batch_size)
        print("VALIDATION | Epoch {}, iter {}: accuracy {}%".format(epoch, train_iter, accuracy))

    model.train()


def train(model, train_data, val_data, args):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(args.max_epochs):
        optimizer.zero_grad()

        train_iter = 0
        for contexts, words, senses in batch_iter(train_data, args.batch_size, shuffle=True):
            # forward pass
            scores = model(contexts, words)
            example_losses = loss_fn(scores, senses)

            batch_loss = example_losses.sum()
            loss = batch_loss / args.batch_size

            # backprop and weight update
            loss.backward()
            # gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            if train_iter % args.print_iter == 0:
                print("Epoch {}, iter {}: loss {}".format(epoch, train_iter, loss))

            if train_iter % args.val_iter == 0:
                validate(model, val_data, epoch, train_iter)

            train_iter += 1
            # TODO add model save and reload

def test(model, test_data, batch_size=128):
    model.eval()

    with torch.no_grad():
        accuracy = evaluate_accuracy(model, test_data, batch_size)
        print("TESTING | Final accuracy {}%".format(accuracy))

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file', default='data/giga5_glv.pkl', type=str)
    parser.add_argument('--corpus_file', default='data/semcor_corpus.pkl', type=str)
    parser.add_argument('--load_corpus_from_file', default=True, type=bool)
    parser.add_argument('--model_name', default='AverageLinear', type=str)
    parser.add_argument('--context_size', default=1, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--max_epochs', default=5, type=int)
    parser.add_argument('--clip_grad', default=5.0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--val_iter', default=500, type=int)
    parser.add_argument('--print_iter', default=200, type=int)
    args = parser.parse_args()

    # get the data
    if args.load_corpus_from_file and os.path.exists(args.corpus_file):
        # load the data from file if it already exists
        with open(args.corpus_file, 'rb') as cf:
            dataset = pickle.load(cf)
            print("Loaded corpus from {}!".format(args.corpus_file))
    else:
        dataset = SemCor(args.context_size)
        print("Parsed corpus!")
        # save the corpus for next time
        with open(args.corpus_file, 'wb') as cf:
            pickle.dump(dataset, cf)
            print("Saved corpus to {} !".format(args.corpus_file))

    # split up the data
    train_data, val_data, test_data = split_data(dataset)

    # load embeddings from file
    emb_weight_matrix_df = pd.read_pickle(args.emb_file)
    emb_weight_matrix = torch.tensor(emb_weight_matrix_df.values)

    # make the model
    if args.model_name == "AverageLinear":
        model = AverageLinear(dataset.max_num_senses, emb_weight_matrix, emb_weight_matrix_df)
    else:
        raise Exception("Invalid model name: {}".format(args.model_name))

    # run the model
    train(model, train_data, val_data, args)
    test(model, test_data)


if __name__ == "__main__":
    main()
