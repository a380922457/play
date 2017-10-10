
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import math
from torch.nn import Parameter

ctx2att = nn.Linear(1024, 512)
p_att_feats = ctx2att(att_feats.view(-1, self.rnn_size)) # B*196*1024
        p_att_feats = p_att_feats.view(*(att_feats.size()[:-1] + (self.att_hid_size,)))