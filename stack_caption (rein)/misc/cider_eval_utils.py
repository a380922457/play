from __future__ import absolute_import
from __future__ import division

from ciderD.ciderD import CiderD
from torch.autograd import Variable
import utils
from cider_dataloader import get_loader
import sys
from collections import OrderedDict
from time import time
reload(sys)
sys.setdefaultencoding("utf-8")


class Evaluator(object):
    def __init__(self):
        self.CiderD_scorer = CiderD()
        self.loader = get_loader(batch_size=1000, shuffle=False, num_workers=1)

    def evaluate(self, model):
        model.eval()
        for i, (images, captions, img_id) in enumerate(self.loader):
            batch_size = len(images)
            images = Variable(images, requires_grad=False).cuda()
            seq, _ = model.sample(images)
            sampled_seq = utils.decode_sequence(seq.cpu())

        res = OrderedDict()
        gts = OrderedDict()

        for i in range(batch_size):
            res[i] = [sampled_seq[i]]
            gts[i] = [captions[i]]

        res = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
        gts = {i: gts[i] for i in range(batch_size)}
        score, _ = self.CiderD_scorer.compute_score(gts, res)
        model.train()
        return score
