from __future__ import absolute_import
from __future__ import division

from torch.autograd import Variable
import utils
from data_loader import get_loader
from pycxtools.coco import COCO
from pycxevalcap.eval import COCOEvalCap
import os
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from time import time

class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.coco = COCO("/media/father/d/ai_challenger_caption_validation_20170910/test100.json")
        self.loader = get_loader(batch_size=100, shuffle=False, num_workers=1, if_train=False)

    def compute_m1(self, json_predictions_file):
        """Compute m1_score"""
        m1_score = {}
        coco_res = self.coco.loadRes(json_predictions_file)
        coco_eval = COCOEvalCap(self.coco, coco_res)
        coco_eval.evaluate()

        for metric, score in coco_eval.eval.items():
            m1_score[metric] = score
        return m1_score

    def language_eval(self, seq, image_id):
        json_predictions_file = './eval_results/result.json'
        data = zip(seq, image_id)
        lines = []
        for seq, image_id in data:
            line = {"caption": seq, "image_id": os.path.splitext(image_id)[0]}
            lines.append(line)
        with open(json_predictions_file, "w") as f:
            json.dump(lines, f)
        m1_score = self.compute_m1(json_predictions_file)
        return m1_score

    def evaluate(self, model, criterion):
        model.eval()
        for i, (images, captions, masks, img_id) in enumerate(self.loader):
            images = Variable(images, requires_grad=False).cuda()
            # captions = Variable(captions, requires_grad=False)
            # torch.cuda.synchronize()
            # captions = captions.cuda()
            # outputs = model(captions, images)
            # loss = criterion(outputs[:, :-1], captions[:, 1:], masks[:, 1:])

            # forward the model to also get generated samples for each image
            seq, _ = model.sample(images)
            decoded_seq = utils.decode_sequence(seq)
            start = time()
            lang_stats = self.language_eval(decoded_seq, img_id)
        model.train()
        return lang_stats  # ,loss.data[0],
