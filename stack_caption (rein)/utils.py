import torch
import torch.nn as nn
import json
from torch.autograd import Variable
decoder_path = '/media/father/d/ai_challenger_caption_20170902/inference_vocab.json'


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(seq):
    with open(decoder_path, "r") as f:
        decoder = json.load(f)
        N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            # stop when meet </S>
            if ix == 2:
                break
            # if ix > 0 :
            if j >= 1:
                txt = txt + ' '
            txt = txt + decoder[str(ix)]
            # else:
            #     break
        out.append(txt)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class Decoder(object):
    def __init__(self):
        super(Decoder, self).__init__()
        with open(decoder_path, "r") as f:
            self.decoder = json.load(f)

    def decode(self, seq):
        txt = ''
        for i in range(len(seq)):
            if seq[i] == 2:
                break
            txt = txt + self.decoder[str(seq[i])]
        return txt


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, outputs0, outputs1, outputs2, target, mask):
        # truncate to the same size
        target = target[:, :outputs0.size(1)]
        mask = mask[:, :outputs0.size(1)]
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        outputs0 = to_contiguous(outputs0).view(-1, outputs0.size(2))
        outputs1 = to_contiguous(outputs1).view(-1, outputs1.size(2))
        outputs2 = to_contiguous(outputs2).view(-1, outputs2.size(2))

        outputs0 = - outputs0.gather(1, target)
        outputs1 = - outputs1.gather(1, target)
        outputs2 = - outputs2.gather(1, target)

        output = (torch.sum(outputs0)+torch.sum(outputs1)+torch.sum(outputs2)) / torch.sum(mask)
        return output


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, outputs1, outputs2, seq1, seq2, reward1, reward2):
        seq1 = seq1.cpu()
        seq2 = seq2.cpu()

        for line in seq1:
            for i, c in enumerate(line):
                if c == 2 and i != 0:
                    if i != len(line)-1:
                        line[:i+1] = 1
                        line[i+1:] = 0
                    else:
                        line[:] = 1
                    break
                else:
                    line[:] = 1
        for line in seq2:
            for i, c in enumerate(line):
                if c == 2 and i != 0:
                    if i != len(line)-1:
                        line[:i+1] = 1
                        line[i+1:] = 0
                    else:
                        line[:] = 1
                    break
                else:
                    line[:] = 1
        mask1 = seq1.float().cuda()
        mask2 = seq2.float().cuda()
        # mask1 = (seq1 > -1).float()
        # mask2 = (seq2 > -1).float()
        # mask2 = to_contiguous(torch.cat([mask2.new(mask2.size(0), 1).fill_(1), mask2[:, :-1]], 1)).view(-1)
        # output1 = - outputs1 * reward1 * Variable(mask1)
        # output1 = - outputs1 * reward1 * Variable(mask1)
        output2 = - outputs2 * (reward1+reward2) * Variable(mask2)
        output = torch.sum(output2)/torch.sum(mask2)
        return output


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
