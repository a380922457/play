# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from time import time
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
from six.moves import cPickle
from data_loader import get_loader
from models.attention_model_v2 import Attention_Model
from utils import LanguageModelCriterion
from eval_utils import Evaluator
import sys
import json
import utils
from collections import Counter
import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding("utf-8")
#
# model = Attention_Model()
# model.cuda()
# criterion = LanguageModelCriterion()
# evaluator = Evaluator()
decoder_path = '/media/father/d/ai_challenger_caption_20170902/inference_vocab.json'
path1 = '/media/father/d/ai_challenger_caption_20170902/train1_5.json'

with open(path1, "r") as f:
    lines = json.load(f)
    for line in lines:
        caption = line["caption"]
        lengths = [len(c) for c in caption]
        print(lengths)




# model.load_state_dict(torch.load(os.path.join("./checkpoint_path/", 'model0.pth')))
#
# val_loss, predictions, lang_stats = evaluator.evaluate(model, criterion)

