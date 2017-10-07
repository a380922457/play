# -*- coding: utf-8 -*-
import os
from PIL import Image
from collections import Counter
from collections import namedtuple
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

image_dir = "/media/father/0000678400004823/ai_challenger_caption_train_20170902/ai_challenger_caption_train_20170902/caption_train_images_20170902"
train_captions_file = '/media/father/0000678400004823/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
output_dir = '/media/father/新加卷/image_feature_att'
word_counts_output_file = '/media/father/0000678400004823/resized_image/vocab.txt'
new_json_file = "/media/father/0000678400004823/ai_challenger_caption_train_20170902/new.json"

start_word = "<S>"
end_word = "</S>"
unknown_word = "<UNK>"

ImageMetadata = namedtuple("ImageMetadata", ["filename", "captions"])
num_threads = 1


def _process_caption(caption):
    tokenized_caption = [start_word]
    tokenized_caption.extend(list(jieba.cut(caption)))
    tokenized_caption.append(end_word)

    return tokenized_caption


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

    word_counts = [x for x in counter.items() if x[1] >= 2]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # with tf.gfile.FastGFile(word_counts_output_file, "w") as f:
    #     f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    # print("Wrote vocabulary file")

    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)

    return vocab


def _process_captions(captions_file):
    with tf.gfile.FastGFile(captions_file, "r") as f:
        caption_data = json.load(f)

    image_metadata = []
    num_captions = 0
    for line in caption_data:
        captions = [_process_caption(c) for c in line["caption"]]
        image_metadata.append(ImageMetadata(line["image_id"], captions))
        num_captions += len(captions)
    print("Finished processing %d captions in %s" % (num_captions, captions_file))
    captions =  [c for image in image_metadata for c in image.captions]
    vocab = _create_vocab(captions)

    new_json = []

    for line in caption_data:
        for caption in line["caption"]:
            caption = _process_caption(caption)
            caption_ids = [vocab.word_to_id(word) for word in caption]
            new_json.append({"image_id":line["image_id"],"caption":caption_ids})
    with open(new_json_file, 'w') as f:
        json.dump(new_json, f)
    # return image_metadata


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
    # _process_captions(train_captions_file)

    _process_images(image_dir)


main()

