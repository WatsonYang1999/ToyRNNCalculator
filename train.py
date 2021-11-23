import torch
from torch.utils.data import DataLoader, random_split

from ToyDataset import ToyDataset, PadSequence
from load_data import load_dataset
from model import Seq2SeqEncoder, Seq2SeqDecoder, EncoderDecoder, MaskedSoftmaxCELoss

datapath = 'data.txt'

# 10 num 4  e
vocab_size = 10 + 1 + 4 + 2 + 3 + 1
embed_dim = 4
hidden_dim = 4
num_layers = 2
batch_size = 50
n_epochs = 10
lr = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_seq2seq(model, n_epochs, optimizer, data_loaders, device):
    losses = []

    def train_epoch(train_or_eval):
        epoch_loss = 0.
        for i, batch in enumerate(data_loaders[train_or_eval]):
            optimizer.zero_grad()
            X, Y, seq_lens = batch

            X = X.to(device)
            Y = Y.to(device)

            Y_hat, _ = model(X, Y)

            l = loss(Y_hat, Y, seq_lens)

            l.sum().backward()
            epoch_loss = l.sum().item()
            optimizer.step()
            losses.append(epoch_loss)

            print(f"Loss :  {epoch_loss} ")

    model.to(device)
    loss = MaskedSoftmaxCELoss()

    for epoch in range(n_epochs):
        train_epoch('train')


if __name__ == '__main__':
    features, labels, seq_lens = load_dataset(datapath)

    toy_set = ToyDataset(features, labels, seq_lens)

    train_size = int(len(toy_set) * 0.8)
    test_size = int(len(toy_set) - train_size)

    train_set, test_set = random_split(toy_set, [train_size, test_size])

    data_loaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=PadSequence()),
                    "test": DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=PadSequence())}

    encoder = Seq2SeqEncoder(vocab_size=vocab_size, embed_size=embed_dim, num_hiddens=hidden_dim, num_layers=num_layers)
    decoder = Seq2SeqDecoder(vocab_size=vocab_size, embed_size=embed_dim, num_hiddens=hidden_dim, num_layers=num_layers)
    net = EncoderDecoder(encoder, decoder)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_seq2seq(net, n_epochs=n_epochs, optimizer=optimizer, data_loaders=data_loaders, device=device)
