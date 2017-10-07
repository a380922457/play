
from PIL import Image
import tensorflow as tf
import json
import os

train_captions_file = '/media/father/0000678400004823/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
image_dir = "/media/father/0000678400004823/resized_image"


with tf.gfile.FastGFile(train_captions_file, "r") as f:
    caption_data = json.load(f)
    for line in caption_data:
        try:
            image = Image.open(os.path.join(image_dir, line["image_id"])).convert('RGB')
        except:
            print("打不开图片", line["url"])
            print(line["image_id"])