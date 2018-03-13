import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable

import torch

from data_utils import SVAgreementCorpus


# TODO EarlyStopping using Callbacks


class LinzenLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dataset):
        super(LinzenLSTM, self).__init__()

        self.word_embeddings = nn.Embedding(dataset.vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)

        self.regression_layer = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, X_batch, lengths, hidden=None):
        embedded_X = self.word_embeddings(X_batch)  # N x max_len x k
        packed_X = self.pad_and_pack(embedded_X, lengths)
        lstm_out, self.hidden = self.lstm(packed_X, hidden)
        h, c = self.hidden
        unnorm_probs = self.regression_layer(h[0])
        return unnorm_probs

    def pad_and_pack(self, X_batch, lengths):
        # pack padded sequence
        X_batch = utils.rnn.pack_padded_sequence(X_batch, lengths, batch_first=True)
        return X_batch


def train(dataset, model, args, debug=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    train_loss = []
    validation_loss = []
    val_accuracies = []

    iterations = len(dataset) / args['batch_size']
    for epoch in range(args['epochs']):
        model.train(mode=True)
        epoch_loss = 0
        for iter in range(iterations):
            optimizer.zero_grad()

            X_batch, y_batch, sents_batch, lengths = dataset.get_batch(data=dataset.data, size=args['batch_size'])

            X_batch = Variable(torch.LongTensor(X_batch), requires_grad=False)  # N x max_len
            y_batch = Variable(torch.LongTensor(y_batch.astype(int)), requires_grad=False)

            y_pred = model(X_batch, lengths)

            loss = criterion(y_pred, target=y_batch)
            epoch_loss += loss.data.numpy()[0]

            if iter % 100 == 0 and debug:
                print 'epoch=', epoch, 'iteration=', iter, 'Loss=', loss.data.numpy()[0]

            loss.backward()
            optimizer.step()
        epoch_loss /= iterations

        # validation
        model.train(mode=False)
        X_val, y_val, _, lengths = dataset.get_batch(data=dataset.val_data, size=len(dataset.val_data))

        X_val = Variable(torch.LongTensor(X_val), requires_grad=False)  # N x max_len
        y_val = Variable(torch.LongTensor(y_val.astype(int)), requires_grad=False)

        y_pred = model.forward(X_val, lengths)

        val_loss = criterion(y_pred, target=y_val).data.numpy()[0]
        val_accu = accuracy(dataset, model, 'val')

        train_loss.append(epoch_loss)
        validation_loss.append(val_loss)
        val_accuracies.append(val_accu)

        if epoch % 7 == 0:
            print 'epoch=', epoch, 'Training Loss=', epoch_loss, 'Val loss=', val_loss, 'Val_accuracy=',

        torch.save(model, './save/' + str(args['lr']) + str(epoch) + '.pt')
    return train_loss, validation_loss, val_accuracies


def accuracy(dataset, model, train_or_val):
    model.train(mode=False)
    if train_or_val == 'train':
        X, y, _, lengths = dataset.get_batch(data=dataset.data, size=len(dataset.data))
    elif train_or_val == 'val':
        X, y, _, lengths = dataset.get_batch(data=dataset.val_data, size=len(dataset.val_data))

    X = Variable(torch.LongTensor(X), requires_grad=False)  # N x max_len
    y = Variable(torch.LongTensor(y.astype(int)), requires_grad=False)

    y_pred = model.forward(X, lengths)

    _, pred = torch.max(y_pred, 1)
    accurate = (pred == y)

    return accurate.data.numpy().mean()


print accuracy(baby_dataset, overfit_model)


if __name__ == '__main__':
    args = {
        'lr': 0.001,
        'epochs': 10,
        'batch_size': 20
    }

    dataset = SVAgreementCorpus()
    model = LinzenLSTM(embedding_dim=50, hidden_dim=50, dataset=dataset)   #dataset

    train(dataset, model, args)