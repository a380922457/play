# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np


class Attention_Model(nn.Module):
    def __init__(self, vocab_size=7767, input_encoding_size=512, rnn_size=1024, seq_length=25,
                  drop_prob_lm=0.5, att_feat_size=2048, att_hid_size=512):
        super(Attention_Model, self).__init__()
        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.drop_prob_lm = drop_prob_lm
        self.seq_length = seq_length
        self.att_feat_size = att_feat_size
        self.att_hid_size = att_hid_size
        self.drop_prob_lm = drop_prob_lm
        self.ss_prob = 0.0

        self.image_linear = nn.Linear(self.att_feat_size, self.rnn_size)
        self.i2h0 = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h0 = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.prev_h2h0 = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout0 = nn.Dropout(self.drop_prob_lm)

        self.embed0 = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.logit0 = nn.Linear(self.rnn_size, self.vocab_size)
        self.a2c0 = nn.Linear(self.att_feat_size, 2 * self.rnn_size)

        ###########################################################
        self.i2h1 = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h1 = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.prev_h2h1 = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout1 = nn.Dropout(self.drop_prob_lm)
        self.h2att1 = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net1 = nn.Linear(self.att_hid_size, 1)

        self.embed1 = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.logit1 = nn.Linear(self.rnn_size, self.vocab_size)
        self.ctx2att1 = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.a2c1 = nn.Linear(self.att_feat_size, 2 * self.rnn_size)

        ###########################################################
        self.i2h2 = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h2 = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.prev_h2h2 = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout2 = nn.Dropout(self.drop_prob_lm)
        self.h2att2 = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net2 = nn.Linear(self.att_hid_size, 1)

        self.embed2 = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.logit2 = nn.Linear(self.rnn_size, self.vocab_size)
        self.ctx2att2 = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.a2c2 = nn.Linear(self.att_feat_size, 2 * self.rnn_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed0.weight.data.uniform_(-initrange, initrange)
        self.logit0.bias.data.fill_(0)
        self.logit0.weight.data.uniform_(-initrange, initrange)

        self.embed1.weight.data.uniform_(-initrange, initrange)
        self.logit1.bias.data.fill_(0)
        self.logit1.weight.data.uniform_(-initrange, initrange)

        self.embed2.weight.data.uniform_(-initrange, initrange)
        self.logit2.bias.data.fill_(0)
        self.logit2.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(bsz, self.rnn_size).zero_()),
                Variable(weight.new(bsz, self.rnn_size).zero_()),
                Variable(weight.new(bsz, self.rnn_size).zero_()),
                Variable(weight.new(bsz, self.rnn_size).zero_()),
                Variable(weight.new(bsz, self.rnn_size).zero_()),
                Variable(weight.new(bsz, self.rnn_size).zero_()))

    def core(self, xt0, xt1, xt2, att_feats, state):
        # layer0
        xt0 = self.embed0(xt0)
        state0 = state[:2]
        att_res = att_feats.mean(2).mean(1)
        coarse_output, state0 = self.lstm0(xt0, state0, att_res, state[4])

        # layer1
        xt1 = self.embed1(xt1)
        att_res = self.attention1(coarse_output, att_feats)
        state1 = state[2:4]
        fine_output, state1 = self.lstm1(xt1, state1, att_res, coarse_output)

        # layer2
        xt2 = self.embed2(xt2)
        att_res = self.attention2(fine_output, att_feats, att_res)
        state2 = state[4:6]
        final_output, state2 = self.lstm2(xt2, state2, att_res, fine_output)

        state = (state0[0], state0[1], state1[0], state1[1], state2[0], state2[1])
        return coarse_output, fine_output, final_output, state

    def lstm0(self, xt, state, att_res, prev_state):

        all_input_sums = self.i2h0(xt) + self.h2h0(state[0]) + self.prev_h2h0(prev_state)

        sigmoid_chunk = F.sigmoid(all_input_sums.narrow(1, 0, 3 * self.rnn_size))
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        cell = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c0(att_res)
        cell = torch.max(cell.narrow(1, 0, self.rnn_size), cell.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1] + in_gate * cell
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout0(next_h)
        state = (next_h, next_c)
        return output, state

    def lstm1(self, xt, state, att_res, prev_state):

        all_input_sums = self.i2h1(xt) + self.h2h1(state[0]) + self.prev_h2h1(prev_state)

        sigmoid_chunk = F.sigmoid(all_input_sums.narrow(1, 0, 3 * self.rnn_size))
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        cell = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c1(att_res)
        cell = torch.max(cell.narrow(1, 0, self.rnn_size), cell.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1] + in_gate * cell
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout1(next_h)
        state = (next_h, next_c)
        return output, state

    def lstm2(self, xt, state, att_res, prev_state):

        all_input_sums = self.i2h2(xt) + self.h2h2(state[0]) + self.prev_h2h2(prev_state)

        sigmoid_chunk = F.sigmoid(all_input_sums.narrow(1, 0, 3 * self.rnn_size))
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        cell = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c2(att_res)
        cell = torch.max(cell.narrow(1, 0, self.rnn_size), cell.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1] + in_gate * cell
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout2(next_h)
        state = (next_h, next_c)
        return output, state

    def attention1(self, state, att_feats):

        p_att_feats = self.ctx2att1(att_feats.view(-1, self.att_feat_size))
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        # 计算attention，获得attention后的图片向量
        att_h = self.h2att1(state)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net1(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size
        weight = F.softmax(dot)  # batch * att_size

        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        return att_res

    def attention2(self, state, att_feats, prev_att_feats):

        p_att_feats = self.ctx2att2(att_feats.view(-1, self.att_feat_size))
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        # 计算attention，获得attention后的图片向量
        state = state + self.image_linear(prev_att_feats)
        att_h = self.h2att2(state)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net2(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size
        weight = F.softmax(dot)  # batch * att_size

        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        return att_res

    def forward(self, seq, att_feats):
        state = self.init_hidden(att_feats.size(0))
        outputs0 = []
        outputs1 = []
        outputs2 = []

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = (torch.DoubleTensor(att_feats.size(0)).uniform_(0, 1)).cuda()
                sample_mask = sample_prob < self.ss_prob

                if sample_mask.sum() == 0:
                    it0 = it1 = it2 = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it0 = seq[:, i].data.clone()
                    it1 = seq[:, i].data.clone()
                    it2 = seq[:, i].data.clone()

                    prob_prev0 = torch.exp(outputs0[-1].data)
                    prob_prev1 = torch.exp(outputs1[-1].data)
                    prob_prev2 = torch.exp(outputs2[-1].data)

                    it0.index_copy_(0, sample_ind, torch.multinomial(prob_prev0, 1).view(-1).index_select(0, sample_ind))
                    it1.index_copy_(0, sample_ind, torch.multinomial(prob_prev1, 1).view(-1).index_select(0, sample_ind))
                    it2.index_copy_(0, sample_ind, torch.multinomial(prob_prev2, 1).view(-1).index_select(0, sample_ind))

                    it0 = Variable(it0, requires_grad=False)
                    it1 = Variable(it1, requires_grad=False)
                    it2 = Variable(it2, requires_grad=False)
                coarse_output, fine_output, final_output, state = self.core(it0, it1, it2, att_feats, state)
            else:
                it = seq[:, i].clone()
                coarse_output, fine_output, final_output, state = self.core(it, it, it, att_feats, state)
            if i >= 1 and seq[:, i].data.sum() == 0:
                break
            output0 = F.log_softmax(self.logit0(coarse_output))
            output1 = F.log_softmax(self.logit1(fine_output))
            output2 = F.log_softmax(self.logit2(final_output))

            outputs0.append(output0)
            outputs1.append(output1)
            outputs2.append(output2)

        return torch.cat([_.unsqueeze(1) for _ in outputs0], 1), torch.cat([_.unsqueeze(1) for _ in outputs1], 1), torch.cat([_.unsqueeze(1) for _ in outputs2], 1)

    def sample(self, att_feats, sample_max=1):
        beam_size = 1
        if beam_size > 1:
            return self.sample_beam(att_feats)

        batch_size = att_feats.size(0)
        state = self.init_hidden(batch_size)

        seq0 = []
        seq1 = []
        seq2 = []

        seq_log_prob1 = []
        seq_log_prob2 = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it2 = (Variable(torch.from_numpy(np.ones(batch_size)).long())).cuda()
            elif sample_max:
                sample_log_prob0, it0 = torch.max(log_prob0.data, 1)
                sample_log_prob1, it1 = torch.max(log_prob1.data, 1)
                sample_log_prob2, it2 = torch.max(log_prob2.data, 1)
                it0 = Variable(it0.view(-1), requires_grad=False).long().cuda()
                it1 = Variable(it1.view(-1), requires_grad=False).long().cuda()
                it2 = Variable(it2.view(-1), requires_grad=False).long().cuda()
            else:
                prob_prev0 = torch.exp(log_prob0.data)  # fetch prev distribution: shape Nx(M+1)
                prob_prev1 = torch.exp(log_prob1.data)
                prob_prev2 = torch.exp(log_prob2.data)
                it0 = Variable(torch.multinomial(prob_prev0, 1), requires_grad=False)
                it1 = Variable(torch.multinomial(prob_prev1, 1), requires_grad=False)
                it2 = Variable(torch.multinomial(prob_prev2, 1), requires_grad=False)

                sample_log_prob1 = log_prob1.gather(1, it1)
                sample_log_prob2 = log_prob2.gather(1, it2)
                it0 = it0.view(-1).long()
                it1 = it1.view(-1).long()
                it2 = it2.view(-1).long()  # and flatten indices for downstream processing
            if t >= 1:
                seq0.append(it0)
                seq1.append(it1)
                seq2.append(it2)

                if not sample_max:
                    seq_log_prob1.append(sample_log_prob1.view(-1))
                    seq_log_prob2.append(sample_log_prob2.view(-1))
            coarse_output, fine_output, final_output, state = self.core(it2, it2, it2, att_feats, state)
            log_prob0 = F.log_softmax(self.logit0(coarse_output))
            log_prob1 = F.log_softmax(self.logit1(fine_output))
            log_prob2 = F.log_softmax(self.logit2(final_output))
        if sample_max:
            return (torch.cat([_.unsqueeze(1) for _ in seq1], 1)).data, (torch.cat([_.unsqueeze(1) for _ in seq2], 1)).data
        else:
            return (torch.cat([_.unsqueeze(1) for _ in seq0], 1)).data, (torch.cat([_.unsqueeze(1) for _ in seq1], 1)).data, (torch.cat([_.unsqueeze(1) for _ in seq2], 1)).data, torch.cat([_.unsqueeze(1) for _ in seq_log_prob1], 1), torch.cat([_.unsqueeze(1) for _ in seq_log_prob2], 1)

    def beam_search(self, state, logprobs, beam_size=3, *args):

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # logprobsf: probabilities augmented after diversity
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = beam_size
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()  # 25 * 3
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()  # 25 * 3
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        beam_size = beam_size

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()  # 25 * 3
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()  # 25 * 3
        beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam  # 3
        done_beams = []

        for t in range(self.seq_length):
            logprobsf = logprobs.data.float()  # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates_divm = beam_step(logprobsf,
                                        beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state)

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            # logprobs, state = self.get_logprobs_state(Variable(it.cuda()), *(args + (state,)))
            logprobs, state = self.get_logprobs_state(Variable(it), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams

    def sample_beam(self, fc_feats, att_feats, beam_size=3):
        beam_size = beam_size
        batch_size = fc_feats.size(0)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.att_feat_size)).view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, self.fc_feat_size)
            tmp_att_feats = att_feats[k:k + 1].expand(*((beam_size,) + att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state)
                logprobs = F.log_softmax(self.logit(output))  # 1 * 10000

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
