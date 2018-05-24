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

train_captions_file = '/media/father/d/ai_challenger_caption_20170902/new_train_val_raw_data.json'
train_image_dir = '/media/father/D74A848338D93A9B/7_train_image_feature_att'
val_image_dir = '/media/father/D74A848338D93A9B/val_image_feature_att'


class MyDataset(data.Dataset):
    def __init__(self,):
        self.image_dir = train_image_dir
        with tf.gfile.FastGFile(train_captions_file, "r") as f:
            self.caption_data = json.load(f)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        line = self.caption_data[index]
        captions = line["caption"]
        img_id = line["image_id"]
        try:
            image = np.load(os.path.join(self.image_dir, str(img_id)) + ".npz")['feat']
        except:
            image = np.load(os.path.join(val_image_dir, str(img_id)) + ".npz")['feat']
        return image, captions, img_id

    def __len__(self):
        return len(self.caption_data)


def collate_fn(data):
    images, target, img_id = zip(*data)
    images = torch.from_numpy(np.array(images))
    return images, target, img_id


def get_loader(batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(dataset=MyDataset(), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
