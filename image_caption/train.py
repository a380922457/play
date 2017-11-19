# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from time import time
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
from data_loader import get_loader
from models.attention_model_v2 import Attention_Model
from utils import LanguageModelCriterion
from eval_utils import Evaluator
import math
import utils


checkpoint_path = '/media/father/c/checkpoint_path'
save_checkpoint_every = 10000
log_step = 100
num_epochs = 30
batch_size = 128
num_workers = 8
init_learning_rate = 4e-4
vocab_size = 7800
use_cuda = True
eval_step = 200


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    data_loader = get_loader(batch_size, shuffle=False, num_workers=num_workers, if_train=True)
    tf_summary_writer = tf.summary.FileWriter(checkpoint_path)

    model = Attention_Model()
    if use_cuda:
        model.cuda()

    criterion = LanguageModelCriterion()
    evaluator = Evaluator()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'optimizer70000.pth')))
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'model70000.pth')))
    model.ss_prob = 0.05
    total_step = len(data_loader)
    iteration = 70001

    total_loss = 0
    for epoch in range(num_epochs):
        epoch += 8
        learning_rate = init_learning_rate*math.pow(0.8, epoch/3)
        if epoch % 5 == 0 and epoch != 0:
            model.ss_prob += 0.05
            model.ss_prob = min(0.25, model.ss_prob)

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        for (images, captions, masks, _) in data_loader:
            images = Variable(images, requires_grad=False)
            captions = Variable(captions, requires_grad=False)
            torch.cuda.synchronize()
            if use_cuda:
                images = images.cuda()
                captions = captions.cuda()
            # Forward, Backward and Optimize
            optimizer.zero_grad()
            outputs = model(captions, images)
            loss = criterion(outputs, captions[:, 1:], masks[:, 1:])
            loss.backward()
            utils.clip_gradient(optimizer, 0.1)

            # train_loss = loss.data[0]
            total_loss += loss.data[0]
            optimizer.step()
            torch.cuda.synchronize()
            if iteration % save_checkpoint_every == 0 and iteration != 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model%d.pth' % iteration))
                print("model saved to {}".format(checkpoint_path))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer%d.pth' % iteration))

            # Print log info
            if iteration % log_step == 0:
                add_summary_value(tf_summary_writer, 'train_loss', total_loss/log_step, iteration)
                endtime = time()
                if iteration != 70100:
                    print('total_time %d, Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (endtime - starttime, epoch, num_epochs, iteration, total_step, total_loss/100, np.exp(loss.data[0])))
                starttime = time()
                total_loss = 0

            if iteration % eval_step == 0 and iteration != 0:
                lang_stats = evaluator.evaluate(model, criterion)

                # add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k, v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()

                add_summary_value(tf_summary_writer, 'learning_rate', learning_rate, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            iteration += 1


if __name__ == '__main__':
    main()
