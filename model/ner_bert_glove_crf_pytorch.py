# Author: Robert Guthrie

import torch
import torch.nn as nn
import numpy as np
import model.transformer as transformer
import model.crf as crf

torch.manual_seed(2019)


# gelu activation function
def gelu(x):
    return x * torch.sigmoid(1.702 * x)


activate_fn_map = {'gelu': gelu, 'sigmoid': nn.functional.sigmoid,
                   'relu': nn.functional.relu, }


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, activate_fn='gelu', bias=True):
        super(Linear, self).__init__()
        self.activate_fn = activate_fn
        self.linear = nn.Linear(dim_in, dim_out, bias)

    def forward(self, feats):
        if self.activate_fn and self.activate_fn in activate_fn_map:
            return activate_fn_map[self.activate_fn](self.linear(feats))
        else:
            return self.linear(feats)


class ConvBlock(nn.Module):
    def __init__(self, in_s, out_s, dil=1, win=3):
        super(ConvBlock, self).__init__()
        pad = dil * (win - 1) // 2
        self.conv1 = nn.Conv1d(in_s, out_s, win, 1, pad, dil)
        self.conv2 = nn.Conv1d(in_s, out_s, win, 1, pad, dil)
        self.use_x = in_s == out_s

    def forward(self, x_in):
        """
        :param x_in: [N, C, L]
        :return: [batch_size, embed_size, seq_len]
        """
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        sigma = torch.sigmoid(x2)
        x_conv = x1 * sigma
        if self.use_x:
            x_conv += x_in
        return x_conv


# Create model
class Bert_CRF(nn.Module):

    def __init__(self, tag_to_ix, params, device):
        super(Bert_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.device = device
        self.tagset_size = len(tag_to_ix)
        self.bert_embeds = None
        self.use_glove = params['use_glove']
        self.mode_type = params['mode_type']
        self.head_num = params['head_num']
        self.te_dropout = params['te_dropout']
        self.use_cross_entropy = params['use_cross_entropy']

        if self.use_glove:
            # glove word embeddings
            glove = np.load(params['glove'])['embeddings']  # np.array
            glove = np.vstack((np.zeros(params['glove_dim']), glove))
            gt = torch.from_numpy(glove).to(device)
            # from_pretrained is class method not object method
            self.glove_embeds = nn.Embedding.from_pretrained(gt)
            self.embedding_dim = params['glove_dim'] + params['bert_dim']
        else:
            self.embedding_dim = params['bert_dim']

        self.hidden_dim = params['hidden_size']

        if self.hidden_dim != self.embedding_dim:
            self.te_linear = Linear(self.embedding_dim, self.hidden_dim)
        else:
            self.te_linear = None

        # select model
        if self.mode_type == 'b':
            self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                                num_layers=1, bidirectional=True)
        elif self.mode_type == 't':
            self.te = transformer.TransformerEncode(self.hidden_dim,
                                                    self.head_num,
                                                    self.te_dropout)
        elif self.mode_type == 'bt':
            self.te1 = transformer.TransformerEncode(self.hidden_dim,
                                                     self.head_num,
                                                     self.te_dropout, False)
            self.te2 = transformer.TransformerEncode(self.hidden_dim,
                                                     self.head_num,
                                                     self.te_dropout, False,
                                                     False)
        elif self.mode_type == 'cnn':
            self.linear = Linear(self.embedding_dim, self.hidden_dim)
            self.conv_1 = ConvBlock(self.hidden_dim, self.hidden_dim)
            self.conv_2 = ConvBlock(self.hidden_dim, self.hidden_dim, dil=2)
            self.conv_4 = ConvBlock(self.hidden_dim, self.hidden_dim, dil=4)

        if self.use_cross_entropy:
            self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        else:
            self.crf = crf.CRF(self.tagset_size, device, False)
        self.dropout_layer = nn.Dropout(params['dropout'])
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

    def init_lstm_params(self, batch_size):
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        h_0 = torch.randn(2, batch_size, self.hidden_dim // 2,
                          device=self.device)
        c_0 = torch.randn(2, batch_size, self.hidden_dim // 2,
                          device=self.device)
        return (h_0, c_0)

    def load_bert_embeds(self, bert_emb_np):
        bt = torch.from_numpy(bert_emb_np).float().to(self.device)
        self.bert_embeds = nn.Embedding.from_pretrained(bt)

    def _get_hidden_features(self, words, words_ids, len_w):
        '''
        :param words: batch of word ids of glove  , batch_size * seq_len
        :param words_ids: batch of word ids of bert  , batch_size * seq_len
        :param len_w: batch of sequence length, batch_size
        :return: lstm result
        '''
        batch_size = words_ids.size()[0]

        # num_layers * num_directions, batch, hidden_size
        self.hidden = self.init_lstm_params(batch_size)
        embeds_bert = self.bert_embeds(words_ids)
        # size batch_size * seq_len * embed_size

        if self.use_glove:
            embeds_glove = self.glove_embeds(words)
            embeds = torch.cat((embeds_glove, embeds_bert), -1)
        else:
            embeds = embeds_bert

        embeds = self.dropout_layer(embeds)

        len_ = len_w.clone()
        mask_x = torch.zeros_like(words_ids, dtype=torch.uint8,
                                  device=self.device)
        for i in range(words_ids.size()[1]):
            mask_x[:, i] = (len_ > 0)
            len_ -= 1
        # change to seq_len first
        mask_x = mask_x.transpose(0, 1)

        if self.mode_type == 'c':
            if self.te_linear:
                embeds = self.te_linear(embeds)

            # only use crf
            return self.hidden2tag(embeds), len_w, mask_x
        elif self.mode_type == 'b':
            # after transpose seq_len,batch_size * embed_size
            embeds = embeds.transpose(0, 1)
            # use bi-lstm
            packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, len_w)
            # input of shape (seq_len, batch, input_size),
            # output size: seq_len, batch, num_directions * hidden_size
            lstm_out, self.hidden = self.lstm(packed, self.hidden)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
                lstm_out)
        elif self.mode_type == 't':
            if self.te_linear:
                embeds = self.te_linear(embeds)

            # use transformer
            te_output = self.te(embeds, mask_x)
            # transpose to seq_len * batch_size * embed_size
            outputs = te_output.transpose(0, 1)
        elif self.mode_type == 'bt':
            if self.te_linear:
                embeds = self.te_linear(embeds)

            te_output = self.te1(embeds, mask_x)
            te_output = self.te2(te_output, mask_x)
            # transpose to seq_len * batch_size * embed_size
            outputs = te_output.transpose(0, 1)
        elif self.mode_type == 'cnn':
            # [N, L, C]
            outputs = self.linear(embeds).transpose(1, 2)
            # [N, C, L]
            outputs = self.conv_1(outputs)
            outputs = self.conv_2(outputs).permute(2, 0, 1)
            # [L, N, C]
            # outputs = self.conv_4(outputs).permute(2, 0, 1)

        # after liner layer: seq_len, batch, tag_size
        hidden_feats = self.hidden2tag(self.dropout_layer(outputs))

        return hidden_feats, len_w, mask_x

    def neg_log_likelihood(self, words, words_ids, len_w, tags):
        feats, len_w, mask_x = self._get_hidden_features(words, words_ids,
                                                         len_w)
        if self.use_cross_entropy:
            seq_len, batch_size, _ = feats.size()
            # change to [batch_size, seq_len, embed_size]
            feats, mask_x = feats.transpose(0, 1), mask_x.transpose(0, 1)
            feats = feats.contiguous().view(-1, self.tagset_size)
            tags = tags.contiguous().view(-1)
            loss = self.cross_entropy(feats, tags).view(batch_size,
                                                        seq_len) * mask_x.float()

            return torch.sum(loss, -1) / len_w
        else:
            # tags = tags.transpose(0, 1)
            # return self.crf.crf_log_loss(feats, tags, mask_x, len_w)
            tags = tags.transpose(0, 1)
            return self.crf.crf_log_loss(feats, tags, mask_x, len_w)

    def forward(self, words, words_ids, len_w):
        # Get the emission scores from the BiLSTM
        feats, len_w, mask_x = self._get_hidden_features(words, words_ids,
                                                         len_w)
        if self.use_cross_entropy:
            # change to [batch_size, seq_len, embed_size]
            feats = feats.transpose(0, 1)
            res = []
            feats_np = torch.argmax(feats, -1).cpu().numpy()
            for i, f_bs in enumerate(feats_np):
                res.append(f_bs[:len_w[i]])
            return np.zeros(feats.size()[0]), res
        else:
            return self.crf(feats, mask_x, len_w)
