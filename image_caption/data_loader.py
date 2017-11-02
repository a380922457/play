# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.utils.data as data
import os
import json
import tensorflow as tf
import numpy as np
from time import time

train_image_dir = '/media/father/c/train_image_feature_att'
val_image_dir = '/media/father/c/val_image_feature_att'
train_captions_file = '/media/father/d/ai_challenger_caption_20170902/train_sorted.json'
val_captions_file = '/media/father/d/ai_challenger_caption_validation_20170910/val1_5.json'


class MyDataset(data.Dataset):
    def __init__(self, if_train):
        self.if_train = if_train
        if if_train:
            self.image_dir = train_image_dir
            with tf.gfile.FastGFile(train_captions_file, "r") as f:
                self.caption_data = json.load(f)[:]
        else:
            self.image_dir = val_image_dir
            with tf.gfile.FastGFile(val_captions_file, "r") as f:
                self.caption_data = json.load(f)[:100]

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        line = self.caption_data[index]
        if self.if_train:
            caption = line["caption"]
        else:
            caption = line["caption"][0]
        img_id = line["image_id"]
        target = torch.Tensor(caption)
        image = np.load(os.path.join(self.image_dir, str(img_id)) + ".npz")['feat']
        return image, target, img_id

    def __len__(self):
        return len(self.caption_data)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, target, img_id = zip(*data)

    images = torch.from_numpy(np.array(images))

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in target]
    targets = torch.zeros(len(target), max(lengths)).long()
    masks = torch.zeros(len(target), max(lengths)).long()
    for i, cap in enumerate(target):
        max_length = max(lengths)
        end = lengths[i]
        targets[i, :end] = cap[:end]
        masks[i, :end] = 1
    return images, targets[:, :max_length], masks[:, :max_length], img_id


def get_loader(batch_size, shuffle, num_workers, if_train):
    return torch.utils.data.DataLoader(dataset=MyDataset(if_train), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
