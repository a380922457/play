import json
import hashlib
import sys
import os

filename = '/media/father/d/ai_challenger_caption_validation_20170910/val1_5.json'
output_name = '/media/father/d/ai_challenger_caption_validation_20170910/test100.json'
decoder_path = '/media/father/d/ai_challenger_caption_20170902/inference_vocab.json'

new_data = {}
annotations = []
images = []

with open(decoder_path, "r") as f:
    decoder = json.load(f)


def decode(line):
    txt = ""
    for j in range(len(line) - 1):
        ix = line[j]
        if j >= 1:
            txt = txt + decoder[str(ix)]
            txt = txt + ' '
    return txt


# dataset = json.load(open("/media/father/d/ai_challenger_caption_validation_20170910/test100.json", "r"))
# lines = dataset["annotations"]
# for line in lines:
#     caption = line["caption"]
#     caption = decode(caption)
#     line["caption"] = "".join(caption)
# dataset["annotations"] = lines

with open(filename, "r") as f:
    data = json.load(f)

for i, line in enumerate(data):
    if i > 99:
        break
    image_id = line["image_id"]
    captions = line["caption"]
    hash_id = int(int(hashlib.sha256(image_id).hexdigest(), 16) % sys.maxint)
    for k, caption in enumerate(captions):
        annotation = {"id": i * 5 + k + 1, "caption": decode(caption), "image_id": hash_id, "file_name": image_id}
        annotations.append(annotation)

        image = {"file_name": os.path.splitext(image_id)[0], "id": hash_id}
        images.append(image)

new_data["annotations"] = annotations
new_data["images"] = images

with open(output_name, "w") as f:
    json.dump(new_data, f)
