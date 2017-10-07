# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
import os
from PIL import Image
import json
import os.path
import tensorflow as tf

image_dir = "/media/father/0000678400004823/resized_image"
captions_file = "/media/father/0000678400004823/ai_challenger_caption_train_20170902/new.json"
word_counts_output_file = '/media/father/0000678400004823/resized_image/vocab.txt'


class MyDataset(data.Dataset):
    def __init__(self, transform=None):
        with tf.gfile.FastGFile(captions_file, "r") as f:
            self.caption_data = json.load(f)
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        line = self.caption_data[index]
        caption = line["caption"]
        img_id = line['image_id']
        target = torch.Tensor(caption)
        image = Image.open(os.path.join(image_dir, img_id))
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.caption_data)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(transform: object, batch_size: object, shuffle: object, num_workers: object) -> object:
    return torch.utils.data.DataLoader(dataset=MyDataset(transform), batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
