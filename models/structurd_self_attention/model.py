import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, embedding_size, hidden_size, d_a,
                 number_attention, vocab_size, number_classes,
                 regularization_coeff, device):
        super(Model, self).__init__()
        self.embedding_size = embedding_size # embedding space dimension
        self.hidden_size = hidden_size # number of hidden units
        self.d_a = d_a # for parameter W_s1
        self.number_attention = number_attention # number of attention parts
        self.vocab_size = vocab_size
        self.regularization_coeff = regularization_coeff
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
        self.linear1 = nn.Linear(2*hidden_size, d_a, bias=False)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(d_a, number_attention)
        self.linear3 = nn.Linear(number_attention * 2 * hidden_size, number_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[-1]
        h0, c0 = self.init_hidden(batch_size)
#         input_tensor = input_tensor.permute(1, 0) # seq_len * batch_size
        x = self.embedding(input_tensor) # seq_len * batch_size * embedding_size
        out, (h, c) = self.lstm(x, (h0, c0)) # seq_len * batch_size * (2 * hidden_size)
        x = self.linear1(out) # seq_len * batch_size * d_a
        x = self.tanh(x) # seq_len * batch_size * d_a
        weights = self.linear2(x) # seq_len * batch_size * number_attention
        A = F.softmax(weights, dim=0) # seq_len * batch_size * number_attention
        A = A.permute(1,2,0) # batch_size * number_attention * seq_len
        out = out.permute(1,0,2) # batch_size * seq_len * (2 * hidden_size)
        M = torch.bmm(A, out) # batch_size * number_attention * (2 * hidden_size)
        M = M.view(batch_size, -1) # batch_size * (number_attention * (2 * hidden_size))
        logits = self.linear3(M) # batch_size * number_classes
        return self.softmax(logits), A 


    def init_hidden(self, batch_size):
        h0 = torch.zeros(size=(2, batch_size, self.hidden_size),
                        device=torch.device(self.device))
        c0  = torch.zeros(size=(2, batch_size, self.hidden_size),
                        device=torch.device(self.device))
        return h0, c0

    def init_model(self):
        pass
