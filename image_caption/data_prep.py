# -*- coding: utf-8 -*-
import os
from PIL import Image
from collections import Counter
from datetime import datetime
import json
import os.path
import jieba
import numpy as np
import tensorflow as tf
from misc.resnet_utils import myResnet
import misc.resnet as resnet
import torch
from torch.autograd import Variable
import sys
from torchvision import transforms as trn

reload(sys)
sys.setdefaultencoding("utf-8")

train_image_dir = '/media/father/d/ai_challenger_caption_20170902/caption_train_images_20170902'
val_image_dir = '/media/father/d/ai_challenger_caption_validation_20170910/caption_validation_images_20170910'

train_captions_file = '/media/father/d/ai_challenger_caption_20170902/caption_train_annotations_20170902.json'
val_captions_file = '/media/father/d/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'

train_new_json_file = '/media/father/d/ai_challenger_caption_20170902/train.json'
val_new_json_file = '/media/father/d/ai_challenger_caption_validation_20170910/val.json'

vocab_file = '/media/father/d/ai_challenger_caption_20170902/vocab.json'
inference_vocab_file = '/media/father/d/ai_challenger_caption_20170902/inference_vocab.json'
image_feature_output_dir = '/media/father/c/train_image_feature_att'
train_att_output_dir = '/media/father/c/train_image_feature_att'
val_att_output_dir = '/media/father/c/val_image_feature_att'

start_word = "<S>"
end_word = "</S>"
unknown_word = "<UNK>"


def _caption_append(caption):
    new_caption = [start_word]
    new_caption.extend(list(jieba.cut(caption)))
    new_caption.append(end_word)
    return new_caption


class Vocabulary(object):
    def __init__(self, vocab, unk_id):
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id


def _create_vocab(captions):
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    word_counts = [x for x in counter.items() if x[1] >= 5]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print word_counts
    print("Words in vocabulary:", len(word_counts))

    # with tf.gfile.FastGFile(word_counts_output_file, "w") as f:
    #     f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    # print("Wrote vocabulary file")

    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    reverse_vocab = dict([(x, y) for (x, y) in enumerate(reverse_vocab)])
    with open(vocab_file, 'w') as f:
        vocab_dict[unknown_word] = unk_id
        json.dump(vocab_dict, f)
    with open(inference_vocab_file, 'w') as f:
        reverse_vocab[unk_id] = unknown_word
        json.dump(reverse_vocab, f)
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab


def _process_captions(captions_file):
    with tf.gfile.FastGFile(captions_file, "r") as f:
        caption_data = json.load(f)

    num_captions = 0
    captions = []
    for line in caption_data:
        caption = [_caption_append(c) for c in line["caption"]]
        num_captions += len(caption)
        for c in caption:
            captions.append(c)
    print("Finished processing %d captions in %s" % (num_captions, captions_file))

    with open('/media/father/d/ai_challenger_caption_20170902/vocab.json', "r") as f:
        decoder = json.load(f)
    vocab = Vocabulary(decoder, len(decoder)-1)
    # vocab = _create_vocab(captions)

    new_json = []

    for line in caption_data:
        captions = []
        for caption in line["caption"]:
            caption = _caption_append(caption)
            caption = [vocab.word_to_id(word)for word in caption]
            captions.append(caption)
        new_json.append({"image_id": line["image_id"], "caption": captions})
    with open(val_new_json_file, 'w') as f:
        json.dump(new_json, f)


preprocess = trn.Compose([
        trn.Normalize([0.475, 0.447, 0.416], [0.257, 0.250, 0.248])
])


def _process_images(image_dir, att_output_dir):
    images = os.listdir(image_dir)

    net = resnet.resnet152(pretrained=True)
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()

    counter = 0
    for image in images:
        try:
            img = Image.open(os.path.join(image_dir, image))
            img = np.array(img).astype("float32")/255.0
            img = torch.from_numpy(img.transpose(2, 0, 1)).cuda()
            img = Variable(preprocess(img), volatile=True)
            tmp_att = my_resnet(img)
            np.savez_compressed(os.path.join(att_output_dir, image), feat=tmp_att.data.cpu().float().numpy())
        except:
            print(image)
        counter += 1
        if not counter % 1000:
            print("%s : Processed %d " % (datetime.now(), counter))


def main():
    # _process_captions(val_image_dir)

    _process_images(val_image_dir, val_att_output_dir)


main()

# def to1_1():
#     with open(path1, "r") as f:
#         lines = json.load(f)
#         new_json = []
#         for line in lines:
#             for caption in line["caption"]:
#                 new_line = {"image_id": line["image_id"], "caption": caption}
#                 new_json.append(new_line)
#
#     with open(path2, "w") as f:
#         json.dump(new_json, f)


