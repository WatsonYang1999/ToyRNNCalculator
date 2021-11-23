import torch
import torch.nn as nn
from torch import Tensor


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.1):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.GRU(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout,
                          batch_first=True)

    def forward(self, X):
        '''

        :param X:
        :return:
        output : [batch_size,seq_len,hidden_dim]
        state  : [num_layers,batch_size,hidden_dim]
        '''

        X = self.embedding(X)

        output, state = self.rnn(X)
        return output, state


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(Seq2SeqDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(input_size=embed_size + num_hiddens, hidden_size=num_hiddens, num_layers=num_layers,
                          dropout=dropout,batch_first=True)

        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs):
        return enc_outputs[1]


    def forward(self, X, state):
        '''

        :param X: [batch_size,seq_len]
        :param state: [embed_size]
        :return:
        '''



        X = self.embedding(X)


        last_state = state[1]

        context = last_state.repeat(X.shape[1],1,1).permute(1,0,2)


        X_and_context = torch.cat((X,context),2)
        output, state = self.rnn(X_and_context,state)
        output = self.dense(output)
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):

        assert torch.Tensor.max(enc_X).item() < 21
        assert torch.Tensor.max(dec_X).item() < 21
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)

    a = torch.arange((maxlen),dtype=torch.float32,device=X.device)[None,:]
    b = valid_len[:,None]

    mask =  a < b





class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):

    def forward(self, input: Tensor, target: Tensor, valid_len) -> Tensor:
        weights = torch.ones_like(target)
        weights = sequence_mask(weights,valid_len)

        self.reduction = 'none'


        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            input.permute(0,2,1),target
        )
        #unweighted_loss = (unweighted_loss * weights).mean(dim = 1)
        return unweighted_loss


if __name__ == '__main__':
    # batch_size = 23
    # seq_len = 7
    # feature_dim = 10
    # embed_dim = 8
    # hidden_dim = 16
    #
    # num_layers = 2
    # # encoder = Seq2SeqEncoder(20, 10, 10, 2)
    # # encoder.eval()
    # # batch = torch.zeros((50, 20),dtype=torch.int)
    # # output, hidden = encoder(batch)
    # # print(output.shape)
    # # print(hidden.shape)
    # encoder = Seq2SeqEncoder(feature_dim, embed_dim, hidden_dim, num_layers)
    # decoder = Seq2SeqDecoder(feature_dim, embed_dim, hidden_dim, num_layers)
    # X = torch.zeros((batch_size, seq_len),dtype=torch.int32)
    # print("X shape:",X.shape)
    # enc_output,enc_hidden = encoder(X)
    # print("Encoder Output :",enc_output.shape)
    # print("Encoder hidden :",enc_hidden.shape)
    # state = decoder.init_state(encoder(X))
    #
    # dec_output,dec_state = decoder(X,state)
    #
    # print("Decoder  output:",dec_output.shape)
    # print("Decoder  state : ",dec_state.shape)
    X = torch.tensor([[1,2,3],[4,5,6]])
    print(sequence_mask(X,torch.tensor(([1,2]))))
