from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.autograd import Variable
import json
import os
import misc.utils as utils
from data_loader import get_loader
from pycxtools.coco import COCO
from pycxevalcap.eval import COCOEvalCap


class Evaluator(object):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.coco = COCO("")
        self.loader = get_loader(batch_size=100, shuffle=False, num_workers=10, if_train=False)

    def compute_m1(self, json_predictions_file):
        """Compute m1_score"""
        m1_score = {}
        coco_res = self.coco.loadRes(json_predictions_file)
        coco_eval = COCOEvalCap(self.coco, coco_res)
        coco_eval.evaluate()

        for metric, score in coco_eval.eval.items():
            print('%s: %.3f' % (metric, score))
            m1_score[metric] = score
        return m1_score

    def language_eval(self, preds):
        json_predictions_file = os.path.join('/eval_results/test.json')

        with open(json_predictions_file, "w") as f:
            json.dump(results, f)

        m1_score = self.compute_m1(json_predictions_file)

        return m1_score

    def eval_split(self, model, criterion):
        model.eval()

        for i, (images, captions, masks) in enumerate(self.loader):
            images = Variable(images, requires_grad=False)
            captions = Variable(captions, requires_grad=False)
            # torch.cuda.synchronize()
            images = images.cuda()
            cuda_captions = captions.cuda()
            outputs = model(captions, images)
            loss = criterion(outputs[:, :-1], cuda_captions[:, 1:], masks[:, 1:])

            # forward the model to also get generated samples for each image
            seq, _ = model.sample(images)

            decoded_seq = utils.decode_sequence(self.loader.get_vocab(), seq)

            lang_stats = self.language_eval(decoded_seq)

        model.train()
        return loss, decoded_seq, lang_stats
