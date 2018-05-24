# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.autograd import Variable
from models import stack_caption_model
import sys
import utils
import torch
import torch.utils.data as data
import os
import json
import numpy as np
from misc.eval_utils import Evaluator

reload(sys)
sys.setdefaultencoding("utf-8")

checkpoint_path0 = '/media/father/D74A848338D93A9B/model11250.pth'
checkpoint_path1 = '/media/father/D74A848338D93A9B/model13125.pth'
checkpoint_path2 = '/media/father/D74A848338D93A9B/123_model9375.pth'
checkpoint_path3 = '/media/father/D74A848338D93A9B/456_model9375.pth'


model0 = stack_caption_model.Attention_Model()
model1 = stack_caption_model.Attention_Model()
model2 = stack_caption_model.Attention_Model()
model3 = stack_caption_model.Attention_Model()


model0.cuda()
model1.cuda()
model2.cuda()
model3.cuda()

test_image_att = '/media/father/D74A848338D93A9B/test_image_feature_BBBB'
json_predictions_file = '/media/father/D74A848338D93A9B/result.json'
lines = []


def language_eval(seq, image_id):
    data = zip(seq, image_id)
    for seq, image_id in data:
        image_id = os.path.splitext(os.path.splitext(image_id)[0])[0]
        line = {"caption": seq, "image_id": image_id}
        lines.append(line)


class MyDataset(data.Dataset):
    def __init__(self):
        self.data = os.listdir(test_image_att)

    def __getitem__(self, index):
        image_id = str(self.data[index])
        try:
            image = np.load(os.path.join(test_image_att, str(self.data[index])))['feat']
        except:
            print(image_id)
        return image, image_id

    def __len__(self):
        return len(self.data)


def evaluate(model1, model2, model3, model4):
    counter = 0
    dataset = data.DataLoader(dataset=MyDataset(), batch_size=128, shuffle=False, num_workers=16)
    for images, image_id in dataset:
        images = Variable(images, requires_grad=False).cuda()
        try:
            seq = []
            batch_size = images.size(0)
            state1 = state2 = state3 = state4 = None
            for t in (range(26)):
                if t == 0:
                    it = (Variable(torch.from_numpy(np.ones(batch_size)).long())).cuda()
                else:
                    log_prob1, state1 = model1.get_logprob(images, it, state1)
                    log_prob2, state2 = model2.get_logprob(images, it, state2)
                    log_prob3, state3 = model3.get_logprob(images, it, state3)
                    log_prob4, state4 = model4.get_logprob(images, it, state4)

                    log_prob = log_prob2 + log_prob1 + log_prob3 + log_prob4
                    _, it = torch.max(log_prob.data, 1)
                    it = Variable(it.view(-1), requires_grad=False).long().cuda()
                if t >= 1:
                    seq.append(it)
            seq = (torch.cat([_.unsqueeze(1) for _ in seq], 1)).data
            decoded_seq = utils.decode_sequence(seq)
            language_eval(decoded_seq, image_id)
        except:
            print(image_id)
        counter += 1
        if counter % 10 == 0:
            print(counter)
    with open(json_predictions_file, "w") as f:
        json.dump(lines, f)

evaluator = Evaluator()

model0.load_state_dict(torch.load(checkpoint_path0))
model1.load_state_dict(torch.load(checkpoint_path1))
model2.load_state_dict(torch.load(checkpoint_path2))
model3.load_state_dict(torch.load(checkpoint_path3))

model0.eval()
model1.eval()
model2.eval()
model3.eval()

evaluate(model0, model1, model2, model3)

