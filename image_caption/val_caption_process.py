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

image_dir = '/media/father/d/ai_challenger_caption_validation_20170910/caption_validation_images_20170910'
val_captions_file = '/media/father/d/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
output_dir = '/media/father/d/ai_challenger_caption_validation_20170910/val_image_att'
word_counts_output_file = '/media/father/d/ai_challenger_caption_validation_20170910/vocab.json'
tokenized_json_file = '/media/father/d/ai_challenger_caption_validation_20170910//tokenized.json'
new_json_file = '/media/father/d/ai_challenger_caption_validation_20170910//new.json'

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
    with open(word_counts_output_file, 'r') as f:
        vocab = json.load(f)
    unk_id = len(vocab)
    vocab = Vocabulary(vocab, unk_id)

    new_json = []
    tokenized_json = []
    for line in caption_data:
        captions = []
        for caption in line["caption"]:
            caption = _caption_append(caption)
            caption_ids = [vocab.word_to_id(word) for word in caption]
            captions.append(caption_ids)
            new_json.append({"image_id": line["image_id"], "caption": caption_ids})
        tokenized_json.append({"image_id": line["image_id"], "captions": captions})
    with open(new_json_file, 'w') as f:
        json.dump(new_json, f)
    with open(tokenized_json_file, 'w') as f:
        json.dump(tokenized_json, f)


def _process_images(image_dir):
    images = os.listdir(image_dir)

    net = resnet.resnet152(pretrained=True)
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()

    counter = 0
    for image in images:
        try:
            img = Image.open(os.path.join(image_dir, image)).resize((224, 224))
            img = np.array(img).astype("float32")/255.0
            img = torch.from_numpy(img.transpose(2, 0, 1)).cuda()
            img = Variable(img, volatile=True)
            tmp_att = my_resnet(img)
            np.savez_compressed(os.path.join(output_dir, image), feat=tmp_att.data.cpu().float().numpy())
        except:
            print(image)
        counter += 1
        if not counter % 1000:
            print("%s : Processed %d " % (datetime.now(), counter))


def main():
    _process_captions(val_captions_file)

    # _process_images(image_dir)


main()

