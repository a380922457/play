# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import utils
import tensorflow as tf
import torch
from torch.autograd import Variable
from cider_dataloader import get_loader
from models.stack_caption_model import Attention_Model
from utils import RewardCriterion
from misc.eval_utils import Evaluator
from rewards import *
import os
from time import time

checkpoint_path = '/media/father/d/rein_models/right_version_123_only_outputs2'
save_checkpoint_every = 1875
log_step = 100
num_epochs = 30
batch_size = 128
num_workers = 8
init_learning_rate = 5e-6
eval_step = 20


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    init_cider_scorer()
    torch.manual_seed(456)
    torch.cuda.manual_seed(456)
    data_loader = get_loader(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tf_summary_writer = tf.summary.FileWriter(checkpoint_path)

    model = Attention_Model()
    model.cuda()

    criterion = RewardCriterion()
    evaluator = Evaluator()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'optimizer11250.pth')))
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'model11250.pth')))
    total_step = len(data_loader)
    iteration = 0

    decoder = utils.Decoder()
    total_loss = 0
    for epoch in range(num_epochs):
        for (images, captions, _) in data_loader:
            images = Variable(images, requires_grad=False).cuda()
            optimizer.zero_grad()
            greedy1, greedy2 = model.sample(images, sample_max=1)
            sampled0, sampled1, sampled2, sample_logprobs1, sample_logprobs2 = model.sample(images, sample_max=0)
            reward1, reward2 = get_self_critical_reward(iteration, captions, sampled0, sampled1, sampled2, greedy1, greedy2, decoder=decoder)

            reward1 = Variable(torch.from_numpy(reward1).float().cuda(), requires_grad=False)
            reward2 = Variable(torch.from_numpy(reward2).float().cuda(), requires_grad=False)
            loss = criterion(sample_logprobs1, sample_logprobs2, sampled1, sampled2, reward1, reward2)
            loss.backward()

            utils.clip_gradient(optimizer, 0.1)
            total_loss += loss.data[0]
            optimizer.step()
            if iteration % save_checkpoint_every == 0 and iteration != 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model%d.pth' % iteration))
                print("model saved to {}".format(checkpoint_path))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer%d.pth' % iteration))

            # Print log info
            if iteration % log_step == 0:
                add_summary_value(tf_summary_writer, 'train_loss', total_loss/log_step, iteration)
                endtime = time()
                if iteration != 0:
                    print('total_time %d, Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (endtime - starttime, epoch, num_epochs, iteration, total_step, total_loss/100))
                starttime = time()
                total_loss = 0

            if iteration % eval_step == 0 and iteration != 0:
                lang_stats = evaluator.evaluate(model)
                for k, v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            iteration += 1

            if iteration == 1:
                evaluator.evaluate(model)


if __name__ == '__main__':
    main()
