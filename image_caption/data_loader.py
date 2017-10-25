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

image_dir = '/media/father/新加卷/image_feature_att'
train_captions_file = '/media/father/新加卷1/ai_challenger_caption_20170902/train.json'
val_captions_file = '/media/father/新加卷/ai_challenger_caption_20170902/val.json'


# image_dir = "/Users/lianghangming/Desktop/image_feature_att"
# captions_file = "/Users/lianghangming/Desktop/new.json"
class MyDataset(data.Dataset):
    def __init__(self, if_train):
        if if_train:
            with tf.gfile.FastGFile(train_captions_file, "r") as f:
                self.caption_data = json.load(f)
        else:
            with tf.gfile.FastGFile(val_captions_file, "r") as f:
                self.caption_data = json.load(f)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        line = self.caption_data[index]
        caption = line["caption"]
        img_id = line['image_id']
        target = torch.Tensor(caption)
        image = np.load(os.path.join(image_dir, str(img_id))+".npz")['feat']

        # except:
        # image = np.load(image_dir+"/0a29e5848947a79fe6236c5d46d290005c69d4c9.jpg.npz")["feat"]
        # target = [1, 3, 3, 4, 5, 6, 2]
        # target = torch.Tensor(target)
        return image, target

    def __len__(self):
        return len(self.caption_data)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.from_numpy(np.array(images))

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    masks = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        masks[i, :end] = 1
    return images, targets, masks


def get_loader(batch_size, shuffle, num_workers, if_train):
    return torch.utils.data.DataLoader(dataset=MyDataset(if_train), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


