import torch
import torch.utils.data as data
import os
import json
import os.path
import tensorflow as tf
import numpy as np
from time import time

image_dir = '/media/father/新加卷1/image_feature_att'
captions_file = '/media/father/新加卷/ai_challenger_caption_20170902/new.json'



# image_dir = "/Users/lianghangming/Desktop/image_feature_att"
# captions_file = "/Users/lianghangming/Desktop/new.json"

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
        image = np.load(os.path.join(image_dir, img_id)+".npz")['feat']

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


def get_loader(batch_size: object, shuffle: object, num_workers: object) -> object:
    return torch.utils.data.DataLoader(dataset=MyDataset(), batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
