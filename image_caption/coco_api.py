import json
import hashlib
import sys
import os

filename = '/media/father/d/ai_challenger_caption_validation_20170910/val.json'
output_name = '/media/father/d/ai_challenger_caption_validation_20170910/test100.json'

with open(filename, "r") as f:
    data = json.load(f)

new_data = {}
annotations = []
images = []
for i, line in enumerate(data):
    if i > 99: break
    image_id = line["image_id"]
    captions = line["caption"]
    hash_id = int(int(hashlib.sha256(image_id).hexdigest(), 16) % sys.maxint)
    for k, caption in enumerate(captions):
        annotation = {"id": i * 5 + k + 1, "caption": caption, "image_id": hash_id, "file_name": image_id}
        annotations.append(annotation)

        image = {"file_name": os.path.splitext(image_id)[0], "id": hash_id}
        images.append(image)

new_data["annotations"] = annotations
new_data["images"] = images

with open(output_name, "w") as f:
    json.dump(new_data, f)
