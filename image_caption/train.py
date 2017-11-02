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
from six.moves import cPickle
from data_loader import get_loader
from models.attention_model_v2 import Attention_Model
from utils import LanguageModelCriterion
from eval_utils import Evaluator
import math
import utils
from models.ada_attention_model import AdaAttModel
from torch.nn import DataParallel

checkpoint_path = "./checkpoint_path/"
model_path = './model_data/'
save_checkpoint_every = 6000
log_step = 100
num_epochs = 20
batch_size = 64
num_workers = 12
init_learning_rate = 5e-4
vocab_size = 7800
use_cuda = True


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    torch.cuda.manual_seed(123)
    data_loader = get_loader(batch_size, shuffle=False, num_workers=num_workers, if_train=True)
    tf_summary_writer = tf.summary.FileWriter(checkpoint_path)

    infos = {}
    histories = {}

    # if start_from is not None:
    #     # open old infos and check if models are compatible
    #     with open(os.path.join(opt.start_from, 'infos_'+'.pkl')) as f:
    #         infos = cPickle.load(f)
    #         saved_model_opt = infos['opt']
    #         need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
    #         for checkme in need_be_same:
    #             assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
    #
    #     if os.path.isfile(os.path.join(opt.start_from, 'histories_'+'.pkl')):
    #         with open(os.path.join(opt.start_from, 'histories_'+'.pkl')) as f:
    #             histories = cPickle.load(f)

    # iteration = infos.get('iter', 0)
    # epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    # loader.iterators = infos.get('iterators', loader.iterators)

    model = AdaAttModel()
    if use_cuda:
        model.cuda()

    # device_ids = range(torch.cuda.device_count())
    # model = DataParallel(model.train(), device_ids)
    criterion = LanguageModelCriterion()
    evaluator = Evaluator()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    # if vars(opt).get('start_from', None) is not None:
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'Ada_optimizer.pth')))
    model.load_state_dict(torch.load(os.path.join("./checkpoint_path/", 'Ada_model24000.pth')))
    total_step = len(data_loader)
    iteration = 24000
    for epoch in range(num_epochs):
        learning_rate = init_learning_rate*math.pow(0.8, epoch)

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
            train_loss = loss.data[0]
            optimizer.step()
            torch.cuda.synchronize()
            if iteration % save_checkpoint_every == 0:
                predictions, lang_stats = evaluator.evaluate(model, criterion)

                # add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k, v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
                val_result_history[iteration] = { 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                current_score = lang_stats['CIDEr']

                best_flag = False
                if True:  # if true
                    # if best_val_score is None or current_score > best_val_score:
                    #     best_val_score = current_score
                    #     best_flag = True
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'Ada_model%d.pth' % iteration))
                    print("model saved to {}".format(checkpoint_path))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'Ada_optimizer.pth'))

                    # Dump miscalleous informations
                    infos['iter'] = iteration
                    infos['epoch'] = epoch
                    # infos['iterators'] = loader.iterators
                    # infos['best_val_score'] = best_val_score

                    histories['val_result_history'] = val_result_history
                    histories['loss_history'] = loss_history
                    histories['lr_history'] = lr_history
                    histories['ss_prob_history'] = ss_prob_history
                    with open(os.path.join(checkpoint_path, 'infos_' + '.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
                    with open(os.path.join(checkpoint_path, 'histories_' + '.pkl'), 'wb') as f:
                        cPickle.dump(histories, f)

                    if best_flag:
                        torch.save(model.state_dict(), os.path.join(checkpoint_path, 'model-best.pth'))
                        print("model saved to {}".format(checkpoint_path))
                        with open(os.path.join(checkpoint_path, 'infos_' + '-best.pkl'), 'wb') as f:
                            cPickle.dump(infos, f)

            # Print log info
            if iteration % log_step == 0:
                endtime = time()
                if iteration != 24000:
                    print('total_time %d, Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (endtime - starttime, epoch, num_epochs, iteration, total_step, loss.data[0], np.exp(loss.data[0])))
                starttime = time()

                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', learning_rate, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

                loss_history[iteration] = train_loss
                ss_prob_history[iteration] = model.ss_prob

            iteration += 1


if __name__ == '__main__':
    main()
