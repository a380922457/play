import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from misc.ciderD import CiderD

CiderD_scorer = None
#CiderD_scorer = CiderD(df='corpus')


def init_cider_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(model, att_feats, captions, gen_result):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = 5
    
    # get greedy decoding baseline
    greedy_res, _ = model.sample(Variable(att_feats.data, volatile=True))

    res = OrderedDict()
    
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(captions)):
        gts[i] = [array_to_str(captions[i][j]) for j in range(len(captions[i]))]

    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    print('Cider scores:', _)

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards
