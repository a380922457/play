from ciderD_scorer import CiderScorer


class CiderD:
    def __init__(self, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma
        self.cider_scorer = CiderScorer(n=self._n)

    def compute_score(self, gts, res):
        self.cider_scorer.clear()
        for res_id in res:
            hypo = res_id['caption']
            ref = gts[res_id['image_id']]
            self.cider_scorer += (hypo[0], ref)

        (score, scores) = self.cider_scorer.compute_score()

        return score, scores

