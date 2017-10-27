# coding: utf-8
import json
import datetime
import jieba
import hashlib
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

class COCO:
    def __init__(self, annotation_file=None):
        self.dataset = {}
        self.anns = []
        self.imgToAnns = {}
        self.imgs = []
        self.image2hash = {}

        if annotation_file is not None:
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, "r"))
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
        anns = {ann['id']: [] for ann in self.dataset['annotations']}

        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']] += [ann]
            anns[ann['id']] = ann

        imgs = {im['id']: {} for im in self.dataset['images']}
        image2hash = {}
        for img in self.dataset['images']:
            imgs[img['id']] = img
            if img['file_name'] in image2hash:
                assert image2hash[img['file_name']] == img['id']
            else:
                image2hash[img['file_name']] = img['id']
        self.image2hash = image2hash

        print('index created!')
        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs

    def getImgIds(self):
        return list(self.imgs.keys())


    def loadRes(self, resFile):
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        # str to hex int for image_id
        imgdict = {}

        def get_image_dict(img_name):
            # image_hash = int(int(hashlib.sha256(img_name).hexdigest(), 16) % sys.maxint)
            image_hash = self.image2hash[img_name]
            if image_hash in imgdict:
                assert imgdict[image_hash] == img_name, 'hash colision: {0}: {1}'.format(image_hash, img_name)
            else:
                imgdict[image_hash] = img_name
            return image_hash

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))

        assert type(anns) == list, 'results in not an array of objects'

        annsImgIds = []
        for ann in anns:
            w = jieba.cut(ann['caption'].strip().replace('。', ''), cut_all=False)
            p = ' '.join(w)
            ann['caption'] = p
            ann['image_id'] = get_image_dict(ann['image_id'])
            annsImgIds.append((ann['image_id']))

        imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
        res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
        for id, ann in enumerate(anns):
            ann['id'] = id
        print('DONE (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res
