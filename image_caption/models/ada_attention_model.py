# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import math
from torch.nn import Parameter
import sys
import numpy as np
from time import time
from torch.nn import DataParallel
reload(sys)
sys.setdefaultencoding("utf-8")


class lstm(nn.Module):
    def __init__(self, input_encoding_size=512, rnn_size=512, att_hid_size=512, drop_prob_lm=0.5):
        super(lstm, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_lm = drop_prob_lm
        self.att_hid_size = att_hid_size

        self.word2h = nn.Linear(self.input_encoding_size, 6 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 6 * self.rnn_size)
        self.image2h = nn.Linear(self.rnn_size, 6 * self.rnn_size)

    def forward(self, xt, state, att_features):
        fc_feature = att_features.mean(1).squeeze()
        hx, cx = state
        gates = (self.word2h(xt) + self.h2h(hx) + self.image2h(fc_feature)).squeeze(0)

        sigmoid_chunk = gates.narrow(1, 0, 4 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        # decode the gates
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        s_gate = sigmoid_chunk.narrow(1, self.rnn_size * 3, self.rnn_size)
        cell = gates.narrow(1, 4 * self.rnn_size, 2 * self.rnn_size)
        cell = torch.max(cell.narrow(1, 0, self.rnn_size), cell.narrow(1, self.rnn_size, self.rnn_size))

        cy = F.tanh((forget_gate * cx) + (in_gate * cell))
        sentinel = s_gate * cy
        hy = out_gate * cy

        hy = F.dropout(hy, self.drop_prob_lm, self.training)
        sentinel = F.dropout(sentinel, self.drop_prob_lm, self.training)
        return hy, cy, sentinel


class AdaAttention(nn.Module):
    def __init__(self, input_encoding_size=512, rnn_size=512, drop_prob_lm=0.5, att_hid_size=512, att_feat_size=2048):
        super(AdaAttention, self).__init__()
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_lm = drop_prob_lm
        self.att_hid_size = att_hid_size
        self.att_feat_size = att_feat_size

        # 图像维度变化
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # 哨兵维度变化
        self.sentinel_linear = nn.Sequential(nn.Linear(self.rnn_size, self.input_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.sentinel_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h隐状态维度变化
        self.ho_linear = nn.Sequential(nn.Linear(self.rnn_size, self.input_encoding_size), nn.Tanh(), nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h, sentinel, att_feats):
        batch_size = att_feats.size(0)
        att_size = 196
        att_feats = att_feats.view(batch_size, att_size, self.rnn_size)
        p_att_feats = (self.ctx2att(att_feats.view(-1, self.rnn_size))).view(batch_size, att_size, self.att_hid_size)

        sentinel_linear = self.sentinel_linear(sentinel)
        sentinel_embd = self.sentinel_embed(sentinel_linear)

        h_linear = self.ho_linear(h)
        h_embed = self.ho_embed(h_linear)

        expanded_h = h_embed.squeeze(0).unsqueeze(1)
        expanded_h = expanded_h.expand(batch_size, att_size + 1, self.att_hid_size)

        img_all = torch.cat([sentinel_linear.view(-1, 1, self.input_encoding_size), att_feats], 1)
        img_all_embed = torch.cat([sentinel_embd.view(-1, 1, self.input_encoding_size), p_att_feats], 1)

        hA = F.dropout(F.tanh(img_all_embed + expanded_h), self.drop_prob_lm, self.training)

        alpha = self.alpha_net(hA.view(-1, self.att_hid_size))
        alpha = F.softmax(alpha.view(-1, att_size + 1))  # B*197

        cHat = torch.bmm(alpha.unsqueeze(1), img_all).squeeze(1)

        atten_out = cHat + h_linear

        h = F.dropout(F.tanh(self.att2h(atten_out)), self.drop_prob_lm, self.training)
        return h.squeeze(0)


class AdaAttModel(nn.Module):
    def __init__(self, vocab_size=7800, input_encoding_size=512, rnn_size=512, drop_prob_lm=0.5, att_feat_size=2048, att_hid_size=512):
        super(AdaAttModel, self).__init__()
        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_lm = drop_prob_lm
        self.att_feat_size = att_feat_size
        self.att_hid_size = att_hid_size
        self.seq_length = 20
        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size, self.input_encoding_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))
        self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size), nn.ReLU(), nn.Dropout(self.drop_prob_lm))

        self.lstm = lstm()
        self.attention = AdaAttention()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, batch_size, self.rnn_size).zero_()),
                Variable(weight.new(1, batch_size, self.rnn_size).zero_()))

    def forward(self,  seq, att_feats):
        state = self.init_hidden(att_feats.size(0))
        outputs = []
        for i in range(seq.size(1)-1):
            it = seq[:, i].clone()
            if i >= 1 and seq[:, i].data.sum() == 0:
                break
            output, state = self.core(it, att_feats, state)

            output = F.log_softmax(self.logit(output))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def sample(self, att_feats, sample_max=1):
        beam_size = 1
        temperature = 1.0
        if beam_size > 1:
            return self.sample_beam(att_feats)

        batch_size = att_feats.size(0)
        state = self.init_hidden(batch_size)

        seq = []
        seq_log_prob = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = (Variable(torch.from_numpy(np.ones(batch_size)).long()))
            elif sample_max:
                sample_log_prob, it = torch.max(log_prob.data, 1)
                it = it.view(-1).long()
                it = Variable(it, requires_grad=False)
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(log_prob.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale log_prob by temperature
                    prob_prev = torch.exp(torch.div(log_prob.data, temperature)).cpu()
                # it = torch.multinomial(prob_prev, 1).cuda()
                it = torch.multinomial(prob_prev, 1)

                sample_log_prob = log_prob.gather(1, Variable(it, requires_grad=False))  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing
                it = Variable(it, requires_grad=False)
            it = it.cuda()
            if t >= 1:
                seq.append(it)  # seq[t] the input of t+2 time step
                seq_log_prob.append(sample_log_prob.view(-1))

            output, state = self.core(it, att_feats, state)
            log_prob = F.log_softmax(self.logit(output))

        return (torch.cat([_.unsqueeze(1) for _ in seq], 1)).data, torch.cat([_.unsqueeze(1) for _ in seq_log_prob], 1)

    def core(self, xt, att_feats, state):
        att_feats = self.att_embed(att_feats.view(att_feats.size(0), -1, self.att_feat_size))
        xt = self.embed(xt)
        hy, cy, sentinel = self.lstm(xt, state, att_feats)
        state = (hy, cy)
        output = self.attention(hy, sentinel, att_feats)
        return output, state

