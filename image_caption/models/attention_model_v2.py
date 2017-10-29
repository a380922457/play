# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np

class Attention_Model(nn.Module):
    def __init__(self, vocab_size=12000, input_encoding_size=300, rnn_size=1024, seq_length=20,
                 num_layers=1, drop_prob_lm=0.5, att_feat_size=2048, att_hid_size=512):
        super(Attention_Model, self).__init__()
        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.drop_prob_lm = drop_prob_lm
        self.seq_length = seq_length
        self.att_feat_size = att_feat_size
        self.att_hid_size = att_hid_size
        self.linear = nn.Linear(self.att_feat_size, self.num_layers * self.rnn_size)  # feature to rnn_size
        self.drop_prob_lm = drop_prob_lm


        # Build a LSTM
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)

        self.init_weights()

    def beam_search(self, state, logprobs, beam_size=3, *args):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
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
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
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

        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
        done_beams = []

        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
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

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))

    def sample_beam(self, fc_feats, att_feats, beam_size=3):
        beam_size = beam_size
        batch_size = fc_feats.size(0)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats.view(-1, self.att_feat_size))
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
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
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def get_log_prob(self, xt, att_feats, state):
        output, state = self.core(xt, att_feats, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)))
        return logprobs, state

    def core(self, xt, att_feats, state):
        xt = self.embed(xt)

        # 投影图像向量，减少计算量
        p_att_feats = self.ctx2att(att_feats.view(-1, self.att_feat_size))
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        # 计算attention，获得attention后的图片向量
        att_h = self.h2att(state[0][-1])  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot)  # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        output, state = self.lstm(xt, state, att_res)
        output = output.squeeze(0)
        return output, state

    def lstm(self, xt, state, att_res):
        # LSTM calculation
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = F.sigmoid(all_input_sums.narrow(1, 0, 3 * self.rnn_size))
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + self.a2c(att_res)
        in_transform = torch.max(in_transform.narrow(1, 0, self.rnn_size),
                                 in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

    def forward(self, seq, att_feats):
        state = self.init_hidden(att_feats.size(0))
        outputs = []
        for i in range(seq.size(1)):
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
                # it = (Variable(torch.from_numpy(np.ones(batch_size)).long())).cuda()
                it = torch.from_numpy(np.ones(batch_size)).long()
            elif sample_max:
                sample_log_prob, it = torch.max(log_prob.data, 1)
                it = it.view(-1).long()
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

            if t >= 1:
                # stop when all finished
                # if t == 1:
                #     unfinished = it > 0
                # else:
                #     unfinished = unfinished * (it > 0)
                # if unfinished.sum() == 0:
                #     break
                # it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step

                seq_log_prob.append(sample_log_prob.view(-1))
            it = Variable(it, requires_grad=False)
            log_prob, state = self.get_log_prob(it, att_feats, state)

        return (torch.cat([_.unsqueeze(1) for _ in seq], 1)).data, torch.cat([_.unsqueeze(1) for _ in seq_log_prob], 1)


