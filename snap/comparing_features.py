
import os
import numpy as np
from utils import load_det_obj_tsv, load_obj_tsv


tsv_frcnn = load_obj_tsv(
    os.path.join('data/mscoco_imgfeat/', 'val2014_obj36.tsv'),
    topk=100)


tsv_frcnn_rosmi = load_obj_tsv(
    os.path.join('data/rosmi/', 'train_obj36.tsv'),
    topk=100)

tsv_rosmi = load_det_obj_tsv(
    os.path.join('data/rosmi/', 'val_obj36.tsv'),
    topk=100)


# print(np.mean(tsv_frcnn_rosmi[0]['features']))
# print(np.mean(tsv_frcnn[1]['features']))
# input(np.mean(tsv_rosmi[1]['features']))
# input(tsv_rosmi[1]['features'].shape)
for id,item in enumerate(tsv_frcnn_rosmi):
    print(item)
    print(item['features'].shape)
    print(tsv_frcnn[id]['features'].shape)
    print(tsv_rosmi[id]['features'].shape)
    print(np.mean(item['features']))
    print(np.mean(tsv_frcnn[id]['features']))
    input(np.mean(tsv_rosmi[id]['features']))
