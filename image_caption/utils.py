import torch
import torch.nn as nn
import json
from time import time
import torch.nn.functional as F

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


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))

        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target)
        output = torch.sum(output) / torch.sum(mask)
        return output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
