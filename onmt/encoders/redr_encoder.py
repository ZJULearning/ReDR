"""Define RNN-based encoders."""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from onmt.encoders.encoder import EncoderBase
from onmt.encoders.rnn_encoder import RNNEncoder


class BaseRNN(nn.Module):
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, input_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.input_size = input_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class EncoderRNN(BaseRNN):
    def __init__(self, input_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru',
                 embedding=None, update_embedding=True):
        super(EncoderRNN, self).__init__(input_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, embedded, input_lengths=None):
        if input_lengths is None:
            output, hidden = self.rnn(embedded)
        else:
            _, idx_sort = torch.sort(input_lengths, dim=0, descending=True)
            _, idx_unsort = torch.sort(idx_sort, dim=0)

            input_lengths = list(input_lengths[idx_sort])

            embedded = self.input_dropout(embedded)
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

            if not self._bidirectional:
                output, hidden = self.rnn(embedded)
            else:
                output, (hidden, _) = self.rnn(embedded)

            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

            output = output.index_select(0, idx_unsort)
            hidden = hidden.index_select(1, idx_unsort)
        return output, hidden


class ReDRLayer(nn.Module):

    def __init__(self, hidden_size, rnn_type, max_len=5000, num_layers=1, n_memory_layers=3):
        super(ReDRLayer, self).__init__()
        self._w_u = Parameter(torch.randn(hidden_size, 1, requires_grad=True))
        self._w_c = Parameter(torch.randn(2 * hidden_size, 1, requires_grad=True))
        self._w_r = Parameter(torch.randn(hidden_size, 1, requires_grad=True))
        self._b = Parameter(torch.randn(1, requires_grad=True))
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.n_memory_layers = n_memory_layers

        self.merge_encoder = EncoderRNN(
            input_size=hidden_size * 3,
            hidden_size=hidden_size,
            n_layers=num_layers,
            rnn_cell=rnn_type,
            max_len=max_len)

    def forward(self, doc_encoder_hidden, que_encoder_hidden, doc_encoder_outputs, que_encoder_outputs):
        # ------------
        if isinstance(doc_encoder_hidden, tuple):
            doc_encoder_hidden = doc_encoder_hidden[0][-1, :, :].unsqueeze(0)
        if isinstance(que_encoder_hidden, tuple):
            que_encoder_hidden = que_encoder_hidden[0][-1, :, :].unsqueeze(0)
        if que_encoder_hidden.size(0) != 1:
            # get the last layer
            que_encoder_hidden = que_encoder_outputs[-1, :, :].unsqueeze(0)
        doc_encoder_outputs = doc_encoder_outputs.transpose(0, 1)
        que_encoder_outputs = que_encoder_outputs.transpose(0, 1)
        batch_size = doc_encoder_outputs.size(0)
        gates = None
        nm_layers = self.n_memory_layers
        # batch * n * d
        _R = doc_encoder_outputs
        # batch * m * d
        _C = que_encoder_outputs
        for i in range(nm_layers):
            # alignment matrix S = R^T C
            # que_encoder_output is context, history C_m
            _S = torch.bmm(_R, _C.transpose(1, 2))
            # _S: B x l_doc x l_que
            # cq: B x hidden_size x l_que
            _H = torch.bmm(_R.transpose(1, 2), torch.softmax(_S, dim=-1))
            _G0 = torch.cat((_C.transpose(1, 2), _H), dim=1)  # B x 2*hidden_size x m
            _G = torch.bmm(_G0, torch.softmax(_S.transpose(1, 2), dim=-1))  # B x 2*hidden_size x n
            u_inp = torch.cat((_R, _G.transpose(1, 2)), dim=-1)  # l_doc x B x 3*hidden_size
            # U: n * batch * hidden
            u_outputs, u_hidden = self.merge_encoder(u_inp)
            # n_layers * direction, batch_size, hidden_size
            encoder_hidden = u_hidden

            if i == 0:
                _R = u_outputs
            else:
                # U, G, R: batch * n * d/2d
                item_a = torch.mm(u_outputs.contiguous().view(-1, self.hidden_size), self._w_u)\
                              .contiguous()\
                              .view(batch_size, -1)
                item_b = torch.mm(_G.transpose(1, 2).contiguous().view(-1, 2 * self.hidden_size), self._w_c)\
                              .contiguous()\
                              .view(batch_size, -1)
                item_c = torch.mm(_R.contiguous().view(-1, self.hidden_size), self._w_r)\
                              .contiguous()\
                              .view(batch_size, -1)
                gates = torch.sigmoid((item_a + item_b + item_c + self._b))
                gates = gates.unsqueeze(-1)
                gated_u = gates * _R + (1 - gates) * u_outputs
                _R = gated_u

        return encoder_hidden, _R


class ReDREncoder(EncoderBase):

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False, n_memory_layers=3):
        super(ReDREncoder, self).__init__()
        self.reference_encoder = RNNEncoder(
                rnn_type, bidirectional, num_layers, hidden_size, dropout, embeddings, use_bridge)
        self.history_encoder = RNNEncoder(
                rnn_type, bidirectional, num_layers, hidden_size, dropout, embeddings, use_bridge)
        self.redr_layer = ReDRLayer(hidden_size, rnn_type, num_layers=num_layers, n_memory_layers=n_memory_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,
            embeddings,
            opt.bridge,
            opt.n_memory_layers)

    def forward(self, src, history, src_lengths=None, history_lengths=None):
        enc_state, memory_bank, src_lengths = self.reference_encoder(src, src_lengths)
        history_enc_state, history_memory_bank, history_lengths = self.history_encoder(history, history_lengths, resort=True)
        enc_hidden, src_enc_outputs = self.redr_layer(enc_state, history_enc_state, memory_bank, history_memory_bank)
        src_enc_outputs = src_enc_outputs.transpose(0, 1)
        return enc_state, src_enc_outputs, src_lengths
