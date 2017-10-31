import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *


class Attention_Model(nn.Module):
    def __init__(self, vocab_size=12000, input_encoding_size=128, rnn_type="lstm", rnn_size=128,
                 num_layers=1, drop_prob_lm=0.5, att_feat_size=2048):
        super(Attention_Model, self).__init__()
        self.vocab_size = vocab_size
        self.input_encoding_size = input_encoding_size
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.drop_prob_lm = drop_prob_lm
        self.att_feat_size = att_feat_size

        self.linear = nn.Linear(self.att_feat_size, self.num_layers * self.rnn_size)  # feature to rnn_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.reduced_feat = nn.Linear(self.att_feat_size, input_encoding_size)

        self.ctx2att = nn.Linear(self.att_feat_size, 1)
        self.h2att = nn.Linear(self.rnn_size, 1)
        self.rnn = nn.LSTM(input_encoding_size*2, rnn_size, num_layers)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
        if self.rnn_type == 'lstm':
            return (image_map, image_map)
        else:
            return image_map

    def forward(self, seq, att_feats):
        state = self.init_hidden(att_feats.mean(2).mean(1).squeeze())
        outputs = []
        for i in range(seq.size(1)):
            it = seq[:, i].clone()
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
            att = att_feats.view(-1, self.att_feat_size)

            att = self.ctx2att(att)  # (batch * att_size) * 1
            att = att.view(-1, att_size)  # batch * att_size
            att_h = self.h2att(state[0][-1])  # batch * 1
            att_h = att_h.expand_as(att)  # batch * att_size
            dot = att_h + att  # batch * att_size

            weight = F.softmax(dot)
            att_feats_ = att_feats.view(-1, att_size, self.att_feat_size)  # batch * att_size * att_feat_size
            att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
            att_res = self.reduced_feat(att_res)

            output, state = self.rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state)
            output = output.squeeze(0)
            output = F.log_softmax(self.logit(self.dropout(output)))
            # output = F.softmax(self.logit(self.dropout(output)))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, self.fc_feat_size)
            tmp_att_feats = att_feats[k:k + 1].expand(*((beam_size,) + att_feats.size()[1:])).contiguous()

            state = self.init_hidden(tmp_fc_feats)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
            done_beams = []
            for t in range(self.seq_length + 1):
                if t == 0:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    """pem a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()  # lets go to CPU for more efficiency in indexing operations
                    ys, ix = torch.sort(logprobsf, 1,
                                        True)  # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c': ix.data[q, c], 'q': q, 'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 1].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 1:
                            beam_seq[:t - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix]
                                                       })

                    # encode as vectors
                    it = beam_seq[t - 1]
                    xt = self.embed(Variable(it.cuda()))

                if t >= 1:
                    state = new_state

                output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output)))

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it,
                                                             requires_grad=False))  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, fc_feats, att_feats, state)
            logprobs = F.log_softmax(self.logit(self.dropout(output)))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
