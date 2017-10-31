import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import math
from torch.nn import Parameter


class lstm(nn.Module):
    def __init__(self, input_encoding_size=300, rnn_size=512, drop_prob_lm=0.5):
        super(lstm, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_lm = drop_prob_lm

        self.w_ih = Parameter(torch.Tensor(5 * self.rnn_size, self.input_encoding_size))
        self.w_hh = Parameter(torch.Tensor(5 * self.rnn_size, self.rnn_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.rnn_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, xt, state):
        hx, cx = state
        gates = (F.linear(xt, self.w_ih) + F.linear(hx, self.w_hh)).squeeze(0)
        in_gate, forget_gate, cell_gate, out_gate, s_gate = gates.chunk(5, 1)

        in_gate = F.sigmoid(in_gate)
        forget_gate = F.sigmoid(forget_gate)
        out_gate = F.sigmoid(out_gate)
        s_gate = F.sigmoid(s_gate)
        cell = F.tanh(cell_gate)

        cy = F.tanh((forget_gate * cx) + (in_gate * cell))
        sentinel = s_gate * cy
        hy = out_gate * cy

        return hy, cy, sentinel


class AdaAttention(nn.Module):
    def __init__(self, input_encoding_size=300, rnn_size=512, drop_prob_lm=0.5, att_hid_size=512, att_feat_size=2048):
        super(AdaAttention, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_lm = drop_prob_lm
        self.att_hid_size = att_hid_size
        self.att_feat_size = att_feat_size

        # 图像维度变化
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # 哨兵维度变化
        # self.sentinel_linear = nn.Sequential(nn.Linear(self.rnn_size, self.input_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.sentinel_embed = nn.Linear(self.rnn_size, self.att_hid_size)

        # h隐状态维度变化
        # self.ho_linear = nn.Sequential(nn.Linear(self.rnn_size, self.input_encoding_size), nn.Tanh(), nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.rnn_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h, sentinel, att_feats):
        batch_size = att_feats.size(0)
        # View into three dimensions
        att_size = att_feats.numel() // batch_size // self.att_feat_size

        # B*196*2048 --> B*196*1024
        att_feats = (self.att_embed(att_feats.view(-1, self.att_feat_size))).view(batch_size, att_size, self.rnn_size)

        # B*196*1024 --> # B*196*512
        att_feats_embd = (self.ctx2att(att_feats.view(-1, self.rnn_size))).view(batch_size, att_size, self.att_hid_size)

        sentinel_embd = self.sentinel_embed(sentinel)  # B*1024 --> B*512

        h_embed = self.ho_embed(h)  # B*512

        expanded_h = h_embed.squeeze(0).unsqueeze(1)
        expanded_h = expanded_h.expand(batch_size, att_size + 1, self.att_hid_size)

        img_all = torch.cat([sentinel.view(-1, 1, self.rnn_size), att_feats], 1)  # B*1*1024 + B*196*1024
        img_all_embed = torch.cat([sentinel_embd.view(-1, 1, self.att_hid_size), att_feats_embd], 1)  # B*197*512

        hA = F.dropout(F.tanh(img_all_embed + expanded_h), self.drop_prob_lm, self.training)

        alpha = self.alpha_net(hA.view(-1, self.att_hid_size))
        alpha = F.softmax(alpha.view(-1, att_size + 1))  # B*197

        cHat = torch.bmm(alpha.unsqueeze(1), img_all).squeeze(1)

        atten_out = cHat + h

        h = F.dropout(F.tanh(self.att2h(atten_out)), self.drop_prob_lm, self.training)
        return h.squeeze(0)


class AdaAttModel(nn.Module):
    def __init__(self, vocab_size=7800, input_encoding_size=300, rnn_size=512, drop_prob_lm=0.5, att_feat_size=2048, att_hid_size=512):
        super(AdaAttModel, self).__init__()
        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_lm = drop_prob_lm
        self.att_feat_size = att_feat_size
        self.att_hid_size = att_hid_size

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        self.lstm = lstm()
        self.attention = AdaAttention()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, batch_size, self.rnn_size).zero_()),
                Variable(weight.new(1, batch_size, self.rnn_size).zero_()))

    def forward(self,  att_feats, seq):
        batch_size = att_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = []

        for i in range(seq.size(1) - 1):
            it = seq[:, i].clone()
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)
            hy, cy, sentinel = self.lstm(xt, state)
            state = (hy, cy)
            output = self.attention(hy, sentinel, att_feats)

            output = F.log_softmax(self.logit(output))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

