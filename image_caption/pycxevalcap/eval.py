from tokenizer.ptbtokenizer import PTBTokenizer
from cider.cider import Cider
from time import time

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        scorer = Cider()
        method = "CIDEr"
        score, scores = scorer.compute_score(gts, res)

        self.eval[method] = score
        self.setImgToEvalImgs(scores, gts.keys(), method)
        print("%s: %0.3f" % (method, score))
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

