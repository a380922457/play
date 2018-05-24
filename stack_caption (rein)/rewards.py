from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from collections import OrderedDict
from misc.ciderD import CiderD
from multiprocessing import Pool
import types
import copy_reg
import pickle
path = '/media/father/73DE59842D44F06B/shuffle_crefs_path/crefs'


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copy_reg.pickle(types.MethodType, _pickle_method)


def init_cider_scorer():
    global scorer
    scorer = CiderD()


def get_self_critical_reward(iteration, captions, sample0, sample1, sample2, greedy1, greedy2, decoder):
    batch_size = sample0.size(0)
    gts = OrderedDict()
    res_sample0 = OrderedDict()
    res_sample1 = OrderedDict()
    res_sample2 = OrderedDict()
    res_greedy1 = OrderedDict()
    res_greedy2 = OrderedDict()
    sampled_seq0 = (sample0.cpu())
    sampled_seq1 = (sample1.cpu())
    sampled_seq2 = (sample2.cpu())
    greedy_seq1 = (greedy1.cpu())
    greedy_seq2 = (greedy2.cpu())

    for i in range(batch_size):
        res_sample0[i] = decoder.decode(sampled_seq0[i, :])
        res_sample1[i] = decoder.decode(sampled_seq1[i, :])
        res_sample2[i] = decoder.decode(sampled_seq2[i, :])
        res_greedy1[i] = decoder.decode(greedy_seq1[i, :])
        res_greedy2[i] = decoder.decode(greedy_seq2[i, :])
        gts[i] = captions[i]

    res_sample0 = [{'image_id': i, 'caption': res_sample0[i]} for i in range(batch_size)]
    res_sample1 = [{'image_id': i, 'caption': res_sample1[i]} for i in range(batch_size)]
    res_sample2 = [{'image_id': i, 'caption': res_sample2[i]} for i in range(batch_size)]
    res_greedy1 = [{'image_id': i, 'caption': res_greedy1[i]} for i in range(batch_size)]
    res_greedy2 = [{'image_id': i, 'caption': res_greedy2[i]} for i in range(batch_size)]

    # crefs, _, score_sample0 = scorer.compute_score(gts, res_sample0, iteration=iteration)
    # print("score_sample0", time()-s)
    iteration = iteration % 1875
    path1 = path + str(iteration)
    with open(path1, "r") as f:
        crefs = pickle.load(f)
    pool = Pool(5)
    a = pool.apply_async(scorer.compute_score, args=(gts, res_sample1, crefs, iteration))
    b = pool.apply_async(scorer.compute_score, args=(gts, res_sample2, crefs, iteration))
    c = pool.apply_async(scorer.compute_score, args=(gts, res_greedy1, crefs, iteration))
    d = pool.apply_async(scorer.compute_score, args=(gts, res_greedy2, crefs, iteration))
    e = pool.apply_async(scorer.compute_score, args=(gts, res_sample0, crefs, iteration))

    pool.close()

    _, _, score_sample1 = a.get()
    _, _, score_sample2 = b.get()
    _, _, score_greedy1 = c.get()
    _, _, score_greedy2 = d.get()
    _, _, score_sample0 = e.get()

    # _, _, score_sample1 = scorer.compute_score(gts, res_sample1, crefs)
    # _, _, score_sample2 = scorer.compute_score(gts, res_sample2, crefs)
    #
    # _, _, score_greedy1 = scorer.compute_score(gts, res_greedy1, crefs)
    # _, _, score_greedy2 = scorer.compute_score(gts, res_greedy2, crefs)

    # print("aaa", np.mean(np.array(score_sample2)))
    # print("bbb", np.mean(np.array(score_greedy2)))

    scores1 = score_sample1 - score_greedy1 + score_sample1 - score_sample0
    scores2 = score_sample2 - score_greedy2 + score_sample2 - score_sample1

    rewards1 = np.repeat(scores1[:, np.newaxis], sampled_seq2.shape[1], 1)
    rewards2 = np.repeat(scores2[:, np.newaxis], sampled_seq2.shape[1], 1)

    return rewards1, rewards2
