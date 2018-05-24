from ciderD_scorer import CiderScorer
import pickle
import os

path = '/media/father/73DE59842D44F06B/shuffle_crefs_path/crefs'


class CiderD:
    def __init__(self, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma
        self.cider_scorer = CiderScorer(n=self._n)

    def compute_score(self, gts, res, import_crefs=None, iteration=None):
        self.cider_scorer.clear()
        # path1 = path + str(iteration)
        # if os.path.exists(path1):
        #     with open(path1, "r") as f:
        #         import_crefs = pickle.load(f)
        for res_id in res:
            hypo = res_id['caption']
            if import_crefs is None:
                ref = gts[res_id['image_id']]
                self.cider_scorer += (hypo, ref)
            else:
                self.cider_scorer += hypo
                self.cider_scorer.crefs = import_crefs
        crefs, score, scores = self.cider_scorer.compute_score()
        # if import_crefs is None:
        #     with open(path1, "w") as f:
        #         pickle.dump(crefs, f)
        # print("cider_scorer.compute_score", time()-s)
        return crefs, score, scores

