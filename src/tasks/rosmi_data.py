# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import *
from lxrt.entry import convert_sents_to_features

from lxrt.tokenization import BertTokenizer
from transformers import BertTokenizer as hBertToken

SCALES = [25,25,4,12,4,4,4]
SCALES2 = [1,1,0.12486,0.49958,0.12486,0.12486,0.12486]
ZOOMS = {
            0:18,
            1:18,
            2:15,
            3:17,
            4:15,
            5:15,
            6:15
        }
GOLD_SIZES = {
            0:25,
            1:25,
            2:3,
            3:12,
            4:3,
            5:3,
            6:3
        }

BEAR2NUMS = {
    "None": -1,
    "North": 0,
    "South": 180,
    "West": 270,
    "East": 90,
    "North West": 315,
    "North East": 45 ,
    "South West": 225,
    "South East": 135

}
#  centers in lat, lon
CENTRES = {
            0:[37.73755663692416, -122.19795016945281],
            1:[32.58577585559755, -117.09164085240766],
            2:[32.61748188924153, -117.14119088106783],
            3:[32.60760476678458, -117.08442647549721],
            4:[37.694753719037756, -122.19294177307802],
            5:[37.71336706451458, -122.19060472858666],
            6:[32.59795016014067, -117.11036626803674]
        }
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000
# Max length including <bos> and <eos>
MAX_SENT_LENGTH = 25
MAX_BOXES = 73
# The path to data and image features.
# VQA_DATA_ROOT = '/scratch/mmk11/data/vqa/'
# IMGFEAT_ROOT = '/scratch/mmk11/data/rosmi/'

class ROSMIDataset:
    """
    ROSMI data example in json file

        {
      "img_id": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5",
      "image_filename": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5.png",
      "scenario_items": "scenario3.json" <--- contains all items of the map
      "landmarks": [
        {
          "name": "husky17",
          "distance": "118",
          "bearing": "0",
          "confidence": "2",
          "raw_gps": [],
          "id": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5_husky17",
          "keywords": "husky robot",
          "g_type": "Point",
          "landmark_gps": [],
          "human_gps": [],
          "landmark_pixels": [ ],
          "human_pixels": [],
          "raw_pixels": []
        }
      ],
      "dynamo_obj": [],
      "gold_coordinates": [],
      "sentid": 279,
      "sentence": {
        "raw": "send husky17 118m in north",
        "imgid": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5",
        "tokens": [    ]
      },
      "gold_pixels": [ ]
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.htokenizer = hBertToken.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(os.path.join(args.data_path,"%s.json" % split))))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['sentid']: datum
            for datum in self.data
        }
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        IMGFEAT_ROOT = args.data_path
        # Loading detection features to img_data
        img_data = []

        img_data.extend(load_det_obj_tsv(
                os.path.join(IMGFEAT_ROOT, 'easy_rosmi_obj36.tsv'),
                topk=topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            c = list(zip(img_datum['t_names'].tolist(), img_datum['t_boxes'].tolist()))
            random.shuffle(c)
            a, b = zip(*c)
            img_datum['t_names'] = np.array(a,dtype='<U100')
            img_datum['t_boxes'] = np.array(b)

            self.imgid2img[img_datum['img_id']] = img_datum

        # Answers
        self.bearing2label = json.load(open(os.path.join(args.data_path,"trainval_bearing2label.json")))
        self.label2bearing = json.load(open(os.path.join(args.data_path,"trainval_label2bearing.json")))
        self.convert2bearing = json.load(open(os.path.join(args.data_path,"convert_bearing_values.json")))
        assert len(self.bearing2label) == len(self.label2bearing)

    @property
    def num_bearings(self):
        return len(self.bearing2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class ROSMITorchDataset(Dataset):
    def __init__(self, dataset: ROSMIDataset):
        super().__init__()
        self.raw_dataset = dataset
        self.max_seq_length = MAX_SENT_LENGTH

        if args.n_ent:
            self.named_entities = True
        else:
            self.named_entities = False
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.htokenizer = hBertToken.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # # Convert img list to dict
        self.imgid2img = self.raw_dataset.imgid2img

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        # with open('val_vocab.json') as training:
        #     train_dict = json.load(training)

        img_id = datum['img_id']
        sent_id = datum['sentid']
        sent = datum['sentence']['raw']

        if datum['landmarks'][0]['g_type'] != 'LineString':
            landmark = torch.tensor(datum['landmarks'][0]['raw_pixels'])
        else:
            landmark = torch.tensor(datum['landmarks'][0]['landmark_pixels'])


        target = torch.tensor(datum['gold_pixels'])



        bearing = torch.zeros(self.raw_dataset.num_bearings)
        bearing[self.raw_dataset.bearing2label[self.raw_dataset.convert2bearing[datum['landmarks'][0]['bearing']]]] = 1

        # start and end id of distance
        tokens = ["[CLS]"] + self.tokenizer.tokenize(sent.strip()) + ["[SEP]"]

        dists = torch.zeros(MAX_SENT_LENGTH)
        diste = torch.zeros(MAX_SENT_LENGTH)
        if datum['landmarks'][0]['distance'] != '0':
            t_distance = self.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
            dists[int(tokens.index(t_distance[0]))]  = 1
            diste[int(tokens.index(t_distance[-1]))]  = 1
        else:
            dists[-1]  = 1
            diste[-1]  = 1


        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        # obj_num = img_info['t_num_boxes']
        feats = img_info['features'].copy()
        # boxes = img_info['boxes'].copy()
        # names = img_info['names'].copy()
        names = img_info['t_names'].copy()
        boxes = img_info['t_boxes'].copy()


        sn_id = int(datum['scenario_items'].split('rio')[1].split('.j')[0])

        centre = calculateTiles(CENTRES[sn_id],ZOOMS[sn_id])

        filename = os.path.join('/home/marios/experiments/gps_prediction/ROSMI/ROSMI_dataset','images', datum["image_filename"])
        landmark_id = 0
        for ipd, name_box in enumerate(names):
            # if datum['landmarks'][0]['g_type'] == 'Point':
            if "".join(datum['landmarks'][0]['name'].split(" ")).lower()  == "".join(name_box[0].split(" ")).lower():
             # or \
             #        int(datum['landmarks'][0]['raw_pixels'][0]) == int(boxes[ipd][0]):
                landmark_id = ipd
                break
            # #     # print(type(datum['landmarks'][0]['raw_pixels']))
            # # #     # print(type(feat_box))
            # # #     # print(datum['landmarks'][0]['raw_pixels'])

                    # tmp_ob = {'g_type':'Point'}
                    # tmp_ob['coordinates'] = datum['landmarks'][0]['raw_gps']
                    # tmp_pixs = generatePixel(tmp_ob,centre,ZOOMS[sn_id],[ 700, 500], 10)
                    #
                    # if tmp_pixs and 'Williams' not in datum['landmarks'][0]['name']:
                    #     px = tmp_pixs["points_x"]
                    #     py = tmp_pixs["points_y"]
                    #     new_bbox = [np.min(px), np.min(py), np.max(px), np.max(py)]
                    #     print(datum['landmarks'][0]['raw_pixels'], boxes[landmark_id], new_bbox)
                    #     print(datum['landmarks'][0]['name'])
                    #     if boxes[landmark_id][0] != datum['landmarks'][0]['raw_pixels'][0]:
                    #         drawItem(['raw_pixels','box_land','new_land'],filename,pixels_bb=[datum['landmarks'][0]['raw_pixels'], list(boxes[landmark_id]), new_bbox])
                    #         input("?")
                #     # input()
                #     # if int(datum['landmarks'][0]['raw_pixels'][0]) == int(feat_box[0]):
                #     #     landmark_id = ipd
            # else:
            #     if "".join(datum['landmarks'][0]['name'].split(" ")).lower()  == "".join(name_box[0].split(" ")).lower() or \
            #             int(datum['landmarks'][0]['landmark_pixels'][0]) == int(boxes[ipd][0]):
            #         landmark_id = ipd
            #         break
            #     print(names)
            #     print(datum['landmarks'][0]['landmark_pixels'], boxes[landmark_id])
            #     print("".join(datum['landmarks'][0]['name'].split(" ")).lower())
            #         # if int(datum['landmarks'][0]['landmark_pixels'][0]) == int(feat_box[0]):
            #         #     landmark_id = ipd
                    # tmp_ob = {'g_type':'Point'}
                    # tmp_ob['coordinates'] = datum['landmarks'][0]['landmark_gps']
                    # tmp_pixs = generatePixel(tmp_ob,centre,ZOOMS[sn_id],[ 700, 500], 10)
                    #
                    # if tmp_pixs and 'Williams' not in datum['landmarks'][0]['name']:
                    #     px = tmp_pixs["points_x"]
                    #     py = tmp_pixs["points_y"]
                    #     new_bbox = [np.min(px), np.min(py), np.max(px), np.max(py)]
                    #     print(datum['landmarks'][0]['landmark_pixels'], boxes[landmark_id], new_bbox)
                    #     print(datum['landmarks'][0]['name'])
                    #     if boxes[landmark_id][0] != datum['landmarks'][0]['landmark_pixels'][0]:
                    #         drawItem(['landmark_pixels','box_land','new_land'],filename,pixels_bb=[datum['landmarks'][0]['landmark_pixels'], list(boxes[landmark_id]), new_bbox])
                    #         input("?")
            #     input("?")

        # last is reserved for landmarks that do not appear in the input feat
        landmark_id_ = torch.zeros(MAX_BOXES)
        if landmark_id == 0:
            landmark_id_[0] = 1
        else:
            landmark_id_[landmark_id] = 1

        feat_mask = 0

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()

        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        feats = torch.from_numpy(feats)
        boxes = torch.from_numpy(boxes)

        _names = 0
        if args.qa:
            map = ""
            for obj_n,obj in enumerate(names):
                map += obj[0]
                if obj_n < len(names) - 1:
                    map += ", "
            # input(map)
            input_ids = self.htokenizer.encode(sent, map)
            # all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            # print(names[landmark_id])
            land_tokens = self.htokenizer.encode(names[landmark_id][0])
            land_tokens.pop(0)
            land_tokens.pop(len(land_tokens)-1)
            # print(land_tokens)
            tmp_lands = input_ids[input_ids.index(102):]


            indices = [i for i, x in enumerate(tmp_lands) if x == land_tokens[0]]
            for ind in indices:
                try:
                    if tmp_lands[ind:ind+len(land_tokens)] == land_tokens:
                        start_index = len(input_ids[:input_ids.index(102)])+ind
                        end_index = start_index + len(land_tokens)
                        break
                except:
                    print("out of list index")
            # print(input_ids[start_index:end_index])

            # input(input_ids)
            token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
            # print(token_type_ids)
            if len(input_ids) > 420:
                input(len(input_ids))

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (420 - len(input_ids))
            input_ids += padding
            input_mask += padding
            token_type_ids += padding

            landmark_start = torch.zeros(420)
            landmark_end = torch.zeros(420)
            # land_end = torch.zeros(420)
            landmark_start[start_index] = 1
            landmark_end[end_index] = 1



            _names = (torch.tensor(input_ids),torch.tensor(token_type_ids),torch.tensor(input_mask))
        else:
            landmark_start = 0
            landmark_end = 0
        # print(input_ids)
        # input(_names[0])


        if self.named_entities:


            names_ids = []
            names_segment_ids = []
            names_mask = []
            for obj in names:
                names_features = convert_sents_to_features(
                    obj, self.max_seq_length, self.tokenizer)

                # for f in names_features
                names_ids.append(torch.tensor(names_features[0].input_ids, dtype=torch.long))
                names_segment_ids.append(torch.tensor(names_features[0].segment_ids, dtype=torch.long))
                names_mask.append(torch.tensor(names_features[0].input_mask, dtype=torch.long))



            if (MAX_BOXES - boxes.shape[0]) > 0:
                feat_mask = torch.ones(boxes.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - boxes.shape[0]), dtype=torch.double)
                feat_mask = torch.cat((feat_mask,feats_padding))
                # Zero-pad up to the sequence length.
                padding = (MAX_BOXES - boxes.shape[0])*[torch.zeros(self.max_seq_length, dtype=torch.long)]

                feats_vis_padding = torch.zeros(((MAX_BOXES - feats.shape[0]),feats.shape[1]), dtype=torch.double)
                box_vis_padding = torch.zeros(((MAX_BOXES - boxes.shape[0]),boxes.shape[1]), dtype=torch.double)
                feats = torch.cat((feats,feats_vis_padding))
                boxes = torch.cat((boxes,box_vis_padding))

                names_ids = torch.stack(names_ids + padding)
                names_segment_ids = torch.stack(names_segment_ids + padding)
                names_mask = torch.stack(names_mask + padding)

                    # bert hidden_size = 768
            else:

                names_ids = torch.stack(names_ids)
                names_segment_ids = torch.stack(names_segment_ids)
                names_mask = torch.stack(names_mask)
                # input(names_ids.shape)
                feat_mask = torch.ones(boxes.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - boxes.shape[0]), dtype=torch.double)
                # # input(feats_padding.shape)
                feat_mask = torch.cat((feat_mask,feats_padding))

            _names = (names_ids, names_segment_ids, names_mask)
        else:
            if (MAX_BOXES - boxes.shape[0]) > 0:
                feat_mask = torch.ones(boxes.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - boxes.shape[0]), dtype=torch.double)
                feat_mask = torch.cat((feat_mask,feats_padding))
                # Zero-pad up to the sequence length.
                # padding = (MAX_BOXES - len(boxes))*[torch.zeros(self.max_seq_length, dtype=torch.long)]

                feats_vis_padding = torch.zeros(((MAX_BOXES - feats.shape[0]),feats.shape[1]), dtype=torch.double)
                box_vis_padding = torch.zeros(((MAX_BOXES - boxes.shape[0]),boxes.shape[1]), dtype=torch.double)
                feats = torch.cat((feats,feats_vis_padding))
                boxes = torch.cat((boxes,box_vis_padding))


            else:
                feat_mask = torch.ones(boxes.shape[0], dtype=torch.double)
                feats_padding = torch.zeros((MAX_BOXES - boxes.shape[0]), dtype=torch.double)
                # # input(feats_padding.shape)
                feat_mask = torch.cat((feat_mask,feats_padding))
            # _names = 0



        return sent_id, feats, feat_mask, boxes, _names, sent,dists, diste,landmark, landmark_id_, bearing,landmark_start,landmark_end, target#bearing
        # return sent_id, feats, feat_mask, boxes, _names, sent,dists, diste,landmark, torch.tensor([landmark_id]), bearing, target#bearing
            # else:
            #     return ques_id, feats, boxes, ques


class ROSMIEvaluator:
    def __init__(self, dataset: ROSMIDataset):
        self.dataset = dataset



    def evaluate(self, sentid2ans: dict):
        target_score = 0.
        meta_score = 0.
        tagging_score = 0.
        meanDist = []
        pixDiff = []
        mDist = 0.
        lands = 0
        counterDist = 0
        thres = 0.50
        # {id:'', sentence:'',gold:[a,b,c],pred:[a,b,c],outcome:True }
        examples = []
        scenarios = {'scenario0.json':[0,0],'scenario1.json':[0,0],'scenario2.json':[0,0],'scenario3.json':[0,0],'scenario4.json':[0,0],'scenario5.json':[0,0],'scenario6.json':[0,0]}
        for sentid, (pred_box, diss,dise, ln, ln_, br, l_s,l_e) in sentid2ans.items():



            siou = 0
            siou3 = 0
            distance2 = None



            datum = self.dataset.id2datum[sentid]
            img_info = self.dataset.imgid2img[datum['img_id']]
            scenarios[datum['scenario_items']][1] += 1
            # obj_num = img_info['num_boxes']
            # # obj_num = img_info['t_num_boxes']
            feats = img_info['features'].copy()
            # boxes = img_info['boxes'].copy()
            # names = img_info['names'].copy()
            boxes = img_info['t_boxes'].copy()
            names = img_info['t_names'].copy()
            sent = datum['sentence']['raw']
            landmark_id_ = 0
            # landmark_id_ = random.randint(0,67)
            for ipd, name_box in enumerate(names):
                if "".join(datum['landmarks'][0]['name'].split(" ")).lower()  == "".join(name_box[0].split(" ")).lower():
                    landmark_id_ = ipd
                    break



            sn_id = int(datum['scenario_items'].split('rio')[1].split('.j')[0])
            # filename = os.path.join('/home/marios/experiments/gps_prediction/ROSMI/ROSMI_dataset','images', datum["image_filename"])
            iou = calc_iou_individual(pred_box, datum['gold_pixels'])
            _scale = 25/SCALES[sn_id]
            siou = iou*_scale
            # iou2 = 1 - iou_loss(pred_box, datum['gold_pixels'])
            # if iou > 0:

            # start and end id of distance
            tokens = ["[CLS]"] + self.dataset.tokenizer.tokenize(datum['sentence']['raw'].strip()) + ["[SEP]"]

            dists = torch.zeros(MAX_SENT_LENGTH)
            diste = torch.zeros(MAX_SENT_LENGTH)
            if datum['landmarks'][0]['distance'] != '0':
                # t_distance = self.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
                t_distance = self.dataset.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())

                start_ = int(tokens.index(t_distance[0]))
                dists[start_]  = 1
                diste[int(tokens[start_:].index(t_distance[-1]))+start_]  = 1
            else:
                dists[-1]  = 1
                diste[-1]  = 1


            dists = np.argmax(dists).item()
            diste = np.argmax(diste).item()
            print("Stats:---------------")
            print(datum['sentence']['raw'])
            print(pred_box,datum['gold_pixels'])
            print(diss,dise, datum['landmarks'][0]['distance'], dists, diste)
            print(br, datum['landmarks'][0]['bearing'])
            print(ln, datum['landmarks'][0]['raw_pixels'])
            try:
                print(f"Landmark ids: {landmark_id_} {names[landmark_id_]} - {ln_} {names[ln_]}")
            except Exception as e:
                print(f"Cannot print stats because {e}")







            centre = calculateTiles(CENTRES[sn_id],ZOOMS[sn_id])

            if landmark_id_ == ln_:
                lands += 1
            try:

                print(boxes[landmark_id_],boxes[ln_])

                pred_cland_coords = getPointLatLng(boxes[ln_][0] + (boxes[ln_][2] - boxes[ln_][0])/2, boxes[ln_][1] + (boxes[ln_][3] - boxes[ln_][1])/2,  \
                                        CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)
            except:
                pred_cland_coords = None

            print(iou, siou)


            pred_coords = getPointLatLng(pred_box[0] + (pred_box[2] - pred_box[0])/2, pred_box[1] +(pred_box[3] - pred_box[1])/2,  \
                            CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)


            # pred_land_coords = getPointLatLng(ln[0] + (ln[2] - ln[0])/2, ln[1] + (ln[3] - ln[1])/2,  \
            #                 CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)

            bearing = BEAR2NUMS[br]
            tmp_pixs2 = None
            final_coord2 = None


            # if datum['landmarks'][0]['distance'] != '0':
                # t_distance = self.dataset.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())

                # if diss == int(tokens.index(t_distance[0])) and dise == int(tokens.index(t_distance[-1])):
            if diss == dists and dise == diste:
                _distance = int(datum['landmarks'][0]['distance'])

                if pred_cland_coords:
                    final_coord2 = destination([pred_cland_coords[1], pred_cland_coords[0]] , _distance, bearing)
                    # final_coord = destination([datum['landmarks'][0]['raw_gps'][0], datum['landmarks'][0]['raw_gps'][1]] , datum['landmarks'][0]['distance'], datum['landmarks'][0]['bearing'])

                    tmp_ob = {'g_type':'Point'}
                    tmp_ob['coordinates'] = final_coord2
                    tmp_pixs2 = generatePixel(tmp_ob,centre,ZOOMS[sn_id],[ 700, 500], GOLD_SIZES[sn_id])

            if final_coord2:
                distance2 = haversine(final_coord2[0],final_coord2[1],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000
                if distance2 < 1:
                    scenarios[datum['scenario_items']][0] += 1

            if distance2:
                mDist += distance2
                distance2 = distance2*SCALES2[sn_id]
                meanDist.append(distance2)

            else:
                counterDist +=1

            print(f"Distance is {distance2}m")


            if tmp_pixs2:
                px = tmp_pixs2["points_x"]
                py = tmp_pixs2["points_y"]
                new_bbox2 = [np.min(px), np.min(py), np.max(px), np.max(py)]

                # try:
                #     img = Image.open(filename)
                # except Exception as e:
                #     print(e)
                #     continue

                prd_center = [new_bbox2[0] + (new_bbox2[2] - new_bbox2[0])/2, new_bbox2[1] + (new_bbox2[3] - new_bbox2[1])/2]
                gold_center = [datum['gold_pixels'][0] + (datum['gold_pixels'][2] - datum['gold_pixels'][0])/2, datum['gold_pixels'][1] + (datum['gold_pixels'][3] - datum['gold_pixels'][1])/2]


                pixDiff.append(sqrt((int(prd_center[1]-gold_center[1]))**2 + (int(prd_center[0]-gold_center[0]))**2))

                iou = calc_iou_individual(new_bbox2, datum['gold_pixels'])
                _scale = 25/SCALES[sn_id]
                # siou3 = iou*_scale
                siou3 = iou/SCALES2[sn_id]
                print(iou*_scale)
                print(siou3)
                # input(iou/SCALES2[datum['scenario_items'].split('rio')[1].split('.json')[0]])
                if siou3 > thres:
                    # print("ONE CORRECT")
                # if ans in label:
                    meta_score += 1
                # drawItem(['gold_pixels','predicted_pixels','landmark'],filename,pixels_bb=[datum['gold_pixels'],new_bbox,ln])

            if siou > thres:
                target_score += 1
            # gold_coords = getPointLatLng(datum['gold_pixels'][0]+GOLD_SIZES[sn_id], datum['gold_pixels'][1]+GOLD_SIZES[sn_id],  \
            #                 CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)
            # print(datum['gold_coordinates'])
            # print(gold_coords)
            # print(haversine(gold_coords[1],gold_coords[0],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000)
            distance = haversine(pred_coords[1],pred_coords[0],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000


            try:
                save_land = str(names[ln_])
            except Exception as e:
                print(f"No examples because {e}")
                save_land = str(None)
            examples.append({ 'id':sentid, 'img_id':datum['img_id'], 'sentence':sent, 'gold':[str(names[landmark_id_]),str(datum['landmarks'][0]['distance'])+' '+str(dists)+ ' '+str(diste),str(datum['landmarks'][0]['bearing'])], 'pred':[save_land,str(diss)+ ' '+str(dise),str(br)], 'outcome': str(siou3 > thres), 'distance':distance2 })


        print(f"Target Score: {target_score / len(sentid2ans)}, Meta Score: {meta_score / len(sentid2ans)}")
        if len(pixDiff) > 0.2*len(sentid2ans):
            # meanD =  mDist / (len(sentid2ans) - counterDist)
            pixMean = int(np.mean(pixDiff))
            # variance = int(np.var(pixDiff))
            pixsd_ = int(np.std(pixDiff))
            distMean = int(np.mean(meanDist))
            # variance = int(np.var(pixDiff))
            distsd_ = int(np.std(meanDist))
        else:
            pixMean = 99999999
            distMean = 99999999
            distsd_ = 99999999
            pixsd_ = 99999999
        print(len(sentid2ans))
        print(lands/len(sentid2ans))
        print(f"Mean distance , Mean pix : {distMean} [{distsd_}] , {pixMean} [{pixsd_}]")
        return target_score / len(sentid2ans), (distMean,distsd_,pixMean,pixsd_,scenarios,examples),tagging_score / len(sentid2ans),meta_score / len(sentid2ans)



class RENCIDataset:
    """
    ROSMI data example in json file

        {
      "scenario_items": "scenario3.json" <--- contains all items of the map
      "landmarks": [
        {
          "name": "husky17",
          "distance": "118",
          "bearing": "0",
          "raw_gps": [],
          "id": "3G5F9DBFOS5RDFXHAP1AIEBZCHJVHO_5_husky17",
          "g_type": "Point"
        }
      ],
      "dynamo_obj": [],
      "gold_coordinates": [],
      "sentid": 279,
      "sentence": {
        "raw": "send husky17 118m in north"
      }
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(os.path.join(args.data_path,"%s.json" % split))))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # making sure no sentence with landmark is being passed
        self.data = [datum for datum in self.data if datum['landmarks'][0]['name']][:1000]
        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['sentid']: datum
            for datum in self.data if datum['landmarks'][0]['name']
        }
        # if args.tiny:
        #     topk = TINY_IMG_NUM
        # elif args.fast:
        #     topk = FAST_IMG_NUM
        # else:
        #     topk = None
        #

        # Load ENC map names and landmarks. Too heavy needs fixing
        IMGFEAT_ROOT = args.data_path
        # with open(os.path.join(IMGFEAT_ROOT,'renci_map.json')) as map:
        #     img_data = json.load(map)
        self.regions = {}
        for scen in [1,3,4,5,7,9,10]:
            with open(os.path.join(IMGFEAT_ROOT,f'scenario{scen}.json')) as map:
                self.regions[f'scenario{scen}.json'] = json.load(map)
        # img_id
        # # Loading detection features to img_data
        # img_data = []
        #
        # img_data.extend(load_det_obj_tsv(
        #         os.path.join(IMGFEAT_ROOT, 'easy_rosmi_obj36.tsv'),
        #         topk=topk))
        # Convert img list to dict
        self.imgid2img = {}
        for datum in self.data:
            tmp_lands = self.regions[datum['scenario_items']] + datum['dynamo_obj']
            random.shuffle(tmp_lands)
            self.imgid2img[datum['img_id']] = tmp_lands
        # input(self.imgid2img)
        # Answers
        self.bearing2label = json.load(open(os.path.join(args.data_path,"trainval_bearing2label.json")))
        self.label2bearing = json.load(open(os.path.join(args.data_path,"trainval_label2bearing.json")))
        self.convert2bearing = json.load(open(os.path.join(args.data_path,"convert_bearing_values.json")))
        assert len(self.bearing2label) == len(self.label2bearing)

    @property
    def num_bearings(self):
        return len(self.bearing2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class RENCITorchDataset(Dataset):
    def __init__(self, dataset: RENCIDataset):
        super().__init__()
        self.raw_dataset = dataset
        self.max_seq_length = MAX_SENT_LENGTH


        self.named_entities = True
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # # Convert img list to dict
        self.imgid2img = self.raw_dataset.imgid2img

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        # with open('val_vocab.json') as training:
        #     train_dict = json.load(training)

        img_id = datum['img_id']
        sent_id = datum['sentid']
        sent = datum['sentence']['raw']

        # if datum['landmarks'][0]['g_type'] != 'LineString':
        #     landmark = torch.tensor(datum['landmarks'][0]['raw_pixels'])
        # else:
        #     landmark = torch.tensor(datum['landmarks'][0]['landmark_pixels'])


        target = torch.tensor(datum['gold_coordinates'])



        bearing = torch.zeros(self.raw_dataset.num_bearings)
        bearing[self.raw_dataset.bearing2label[self.raw_dataset.convert2bearing[str(datum['landmarks'][0]['bearing'])]]] = 1

        # start and end id of distance
        tokens_a =  self.tokenizer.tokenize(sent.strip())
        # print(tokens_a)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > MAX_SENT_LENGTH - 2:
            tokens_a = tokens_a[:(MAX_SENT_LENGTH - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        # print(tokens)
        dists = torch.zeros(MAX_SENT_LENGTH)
        diste = torch.zeros(MAX_SENT_LENGTH)
        if datum['landmarks'][0]['distance'] != '0':
            t_distance = self.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
            start_ = int(tokens.index(t_distance[0]))
            dists[start_]  = 1
            diste[int(tokens[start_:].index(t_distance[-1]))+start_]  = 1
        else:
            dists[-1]  = 1
            diste[-1]  = 1

        # sentence taggin for landmarks.
        land_s = torch.zeros(MAX_SENT_LENGTH)
        land_e = torch.zeros(MAX_SENT_LENGTH)
        t_name = self.tokenizer.tokenize(datum['landmarks'][0]['name'].strip())
        # print(t_name)
        # print(datum['landmarks'][0]['name'])
        # print(sent)
        start_ = [idx for idx,x in enumerate(tokens) if t_name[0] in x][0]
        land_s[start_]  = 1
        # land_e[int(tokens.index(t_name[0])) + len(t_name)-1]  = 1
        land_e[start_ + len(t_name)-1]  = 1





        # Get image info
        img_info = self.imgid2img[img_id]
        # img_info = datum['dynamo_obj'] + self.regions[temp_enc['scenario_items']]


        # obj_num = img_info['num_boxes']
        # obj_num = img_info['t_num_boxes']
        # feats = img_info['features'].copy()
        # boxes = img_info['boxes'].copy()
        # names = img_info['names'].copy()
        # names = img_info['t_names'].copy()
        # boxes = img_info['coordinates'].copy()
        # boxes = img_info['t_boxes'].copy()
        # print(img_info)
        names = [x['name'] for x in img_info]
        boxes = [x['coordinates'] for x in img_info]
        ids = [x['id'] for x in img_info]
        # print(names[0])
        # print(boxes[0])
        # input(img_info[0])


        landmark_id = 0
        for ipd, name_box in enumerate(ids):
            if datum['landmarks'][0]['id'] == name_box:
                landmark_id = ipd
                break
            # if datum['landmarks'][0]['g_type'] == 'Point':
            # if "".join(datum['landmarks'][0]['name'].split(" ")).lower()  == "".join(name_box[0].split(" ")).lower():
            #  # or \
            #  #        int(datum['landmarks'][0]['raw_pixels'][0]) == int(boxes[ipd][0]):
            #     landmark_id = ipd
            #     break

        # last is reserved for landmarks that do not appear in the input feat
        landmark_id_ = torch.zeros(MAX_BOXES)
        if landmark_id == 0:
            landmark_id_[0] = 1
        else:
            landmark_id_[landmark_id] = 1


        _names = 0



        if self.named_entities:

            names_ids = []
            names_segment_ids = []
            names_mask = []
            # print(names)
            for obj in names:
                # print(obj)
                names_features = convert_sents_to_features(
                    obj, self.max_seq_length, self.tokenizer)

                # for f in names_features
                names_ids.append(torch.tensor(names_features[0].input_ids, dtype=torch.long))
                names_segment_ids.append(torch.tensor(names_features[0].segment_ids, dtype=torch.long))
                names_mask.append(torch.tensor(names_features[0].input_mask, dtype=torch.long))


            # print(len(names_ids))
            padding = (73 - len(names_ids))*[torch.zeros(MAX_SENT_LENGTH, dtype=torch.long)]

            names_ids = torch.stack(names_ids + padding)
            names_segment_ids = torch.stack(names_segment_ids + padding)
            names_mask = torch.stack(names_mask + padding)
            # print(names_ids.shape)

            # pseudo values
            feats = torch.zeros(len(names_ids),2048)
            feat_mask = torch.ones(len(names_ids), dtype=torch.double)
            feats_padding = torch.zeros((73 - len(names_ids)), dtype=torch.double)
            feat_mask = torch.cat((feat_mask,feats_padding))
            boxes = torch.zeros(len(names_ids),4)
            landmark = torch.zeros(4)
            # landmark_start = 0
            # landmark_end = 0
            _names = (names_ids, names_segment_ids, names_mask)



        # diss = np.argmax(dists).item()
        # dise = np.argmax(diste).item()
        # lan_s = np.argmax(land_s).item()
        # lan_e = np.argmax(land_e).item()
        # print("Stats:---------------")
        # print(sent)
        # print(datum['landmarks'][0]['distance'], diss, dise, tokens[diss :dise+1])
        # print(bearing, datum['landmarks'][0]['bearing'])
        # print(f"land :{tokens[lan_s:lan_e+1]}, {lan_s},{lan_e}")
        # print(f"Landmark ids: {landmark_id} {names[landmark_id]}")
        # input("?")

        return sent_id, feats, feat_mask, boxes, _names, sent,dists, diste,landmark, landmark_id_, bearing,land_s,land_e, target#bearing


class RENCIEvaluator:
    def __init__(self, dataset: RENCIDataset):
        self.dataset = dataset



    def evaluate(self, sentid2ans: dict):
        target_score = 0.
        meta_score = 0.
        tagging_score = 0.
        meanDist = []
        pixDiff = []
        mDist = 0.
        lands = 0
        counterDist = 0
        thres = 0.50
        # {id:'', sentence:'',gold:[a,b,c],pred:[a,b,c],outcome:True }
        examples = []
        scenarios = {'scenario0.json':[0,0],'scenario1.json':[0,0],'scenario2.json':[0,0],'scenario3.json':[0,0],'scenario4.json':[0,0],'scenario5.json':[0,0],'scenario6.json':[0,0],'scenario7.json':[0,0],'scenario8.json':[0,0],'scenario9.json':[0,0],'scenario10.json':[0,0]}
        for sentid, (pred_box, diss,dise, ln, ln_, br, l_s,l_e) in sentid2ans.items():


            siou3 = 0
            distance2 = None

            datum = self.dataset.id2datum[sentid]
            img_info = self.dataset.imgid2img[datum['img_id']]
            scenarios[datum['scenario_items']][1] += 1
            names = [x['name'] for x in img_info]
            ids = [x['id'] for x in img_info]
            boxes = [x['coordinates'] for x in img_info]


            sent = datum['sentence']['raw']
            landmark_id_ = 0
            for ipd, name_box in enumerate(ids):
                if datum['landmarks'][0]['id'] == name_box:
                    landmark_id_ = ipd
                    break

            # start and end id of distance
            tokens_a =  self.dataset.tokenizer.tokenize(datum['sentence']['raw'].strip())
            # print(tokens_a)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > MAX_SENT_LENGTH - 2:
                tokens_a = tokens_a[:(MAX_SENT_LENGTH - 2)]

            # Keep segment id which allows loading BERT-weights.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

            # sentence taggin for landmarks.
            land_s = torch.zeros(MAX_SENT_LENGTH)
            land_e = torch.zeros(MAX_SENT_LENGTH)
            t_name = self.dataset.tokenizer.tokenize(datum['landmarks'][0]['name'].strip())
            start_ = [idx for idx,x in enumerate(tokens) if t_name[0] in x][0]
            land_s[start_]  = 1
            # land_e[int(tokens.index(t_name[0])) + len(t_name)-1]  = 1
            land_e[start_ + len(t_name)-1]  = 1

            dists = torch.zeros(MAX_SENT_LENGTH)
            diste = torch.zeros(MAX_SENT_LENGTH)
            if datum['landmarks'][0]['distance'] != '0':
                # t_distance = self.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
                t_distance = self.dataset.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())
                start_ = int(tokens.index(t_distance[0]))
                dists[start_]  = 1
                diste[int(tokens[start_:].index(t_distance[-1]))+start_]  = 1
            else:
                dists[-1]  = 1
                diste[-1]  = 1

            dists = np.argmax(dists).item()
            diste = np.argmax(diste).item()
            land_s = np.argmax(land_s).item()
            land_e = np.argmax(land_e).item()
            try:
                print("Stats:---------------")
                print(datum['sentence']['raw'])
                print(diss,dise, datum['landmarks'][0]['distance'], dists, diste)
                print(br, datum['landmarks'][0]['bearing'])
                print(f"land :{l_s}, {l_e}, {tokens[l_s:l_e+1]}, {land_s},{land_e}")
                print(f"Landmark ids: {landmark_id_} {names[landmark_id_]} - {ln_} {names[ln_]}")
            except Exception as e:
                print(f"Cannot print stats because {e}")



            #
            if landmark_id_ == ln_:
                lands += 1
                meta_score += 1
            # try:
            #
            #     print(boxes[landmark_id_],boxes[ln_])
            #     pred_cland_coords = [np.mean([x[0] for x in boxes[ln_]['coordinates']]),np.mean([x[1] for x in boxes[ln_]['coordinates']])]
            #     #
            #     # pred_cland_coords = getPointLatLng(boxes[ln_][0] + (boxes[ln_][2] - boxes[ln_][0])/2, boxes[ln_][1] + (boxes[ln_][3] - boxes[ln_][1])/2,  \
            #     #                         CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)
            # except:
            #     pred_cland_coords = None
            #


            bearing = BEAR2NUMS[br]
            tmp_pixs = None
            tmp_pixs2 = None
            final_coord2 = None

            pred_cland_coords = None

            # if datum['landmarks'][0]['distance'] != '0':
                # t_distance = self.dataset.tokenizer.tokenize(datum['landmarks'][0]['distance'].strip())

                # if diss == int(tokens.index(t_distance[0])) and dise == int(tokens.index(t_distance[-1])):
            if diss == dists and dise == diste and (landmark_id_ == ln_ or (l_s == land_s and l_e == land_e)):


                lands += 1
                tagging_score += 1
                siou3 = 100
                try:

                    # print(boxes[landmark_id_],boxes[ln_])
                    # input()
                    pred_cland_coords = [np.mean([x[0] for x in boxes[ln_]]),np.mean([x[1] for x in boxes[ln_]])]
                    #
                    print(pred_cland_coords)
                    # pred_cland_coords = getPointLatLng(boxes[ln_][0] + (boxes[ln_][2] - boxes[ln_][0])/2, boxes[ln_][1] + (boxes[ln_][3] - boxes[ln_][1])/2,  \
                    #                         CENTRES[sn_id][1],CENTRES[sn_id][0],ZOOMS[sn_id], 500, 700)
                except Exception as e:
                    print(e)
                    pred_cland_coords = None
                print("correct")
                # print(pred_cland_coords)
                _distance = int(datum['landmarks'][0]['distance'])


                if pred_cland_coords:
                    final_coord2 = destination([pred_cland_coords[0], pred_cland_coords[1]] , _distance, bearing)

                    # final_coord = destination([datum['landmarks'][0]['raw_gps'][0], datum['landmarks'][0]['raw_gps'][1]] , datum['landmarks'][0]['distance'], datum['landmarks'][0]['bearing'])
                    # print(f"Final coord {final_coord2}")
                    tmp_ob = {'g_type':'Point'}
                    tmp_ob['coordinates'] = final_coord2

            # else:
            #     input("Wrong!!! ")


            if final_coord2:
                print(final_coord2,datum['gold_coordinates'])
                distance2 = haversine(final_coord2[0],final_coord2[1],datum['gold_coordinates'][0],datum['gold_coordinates'][1])*1000
                if distance2 < 1:
                    scenarios[datum['scenario_items']][0] += 1
                print(f"Distance is {distance2}m")

                #
                #
                # if siou3 > thres:
                #     # print("ONE CORRECT")
                # # if ans in label:
                #     score3 += 1
                # drawItem(['gold_pixels','predicted_pixels','landmark'],filename,pixels_bb=[datum['gold_pixels'],new_bbox,ln])


            if distance2 is not None:
                mDist += distance2
                meanDist.append(distance2)

            else:
                counterDist +=1
            try:
                save_land = str(names[ln_])
            except Exception as e:
                print(f"No examples because {e}")
                save_land = str(None)
            examples.append({ 'id':sentid, 'img_id':datum['img_id'], 'sentence':sent, 'gold':[str(names[landmark_id_]),str(datum['landmarks'][0]['distance'])+' '+str(dists)+ ' '+str(diste),str(datum['landmarks'][0]['bearing'])], 'pred':[save_land,str(diss)+ ' '+str(dise),str(br), tokens[l_s:l_e+1]], 'outcome': str(siou3 > thres), 'distance':distance2 })

        print(f" Target Score: {target_score / len(sentid2ans)}, MetaData Score: {meta_score / len(sentid2ans)}, Tagging Score {tagging_score / len(sentid2ans)}")
        # if len(pixDiff) > 0.2*len(sentid2ans):
        #     # meanD =  mDist / (len(sentid2ans) - counterDist)
        #     pixMean = int(np.mean(pixDiff))
        #     # variance = int(np.var(pixDiff))
        #     pixsd_ = int(np.std(pixDiff))
        # print(meanDist)
        if len(meanDist) > 0:
            distMean = int(np.mean(meanDist))
        #     # variance = int(np.var(pixDiff))
            distsd_ = int(np.std(meanDist))
            pixMean = 99999999
            pixsd_ = 99999999
        else:
            pixMean = 99999999
            distMean = 99999999
            distsd_ = 99999999
            pixsd_ = 99999999
        print(len(sentid2ans))
        print(lands/len(sentid2ans))
        print(f"Mean distance , Mean pix : {distMean} [{distsd_}] , {pixMean} [{pixsd_}]")

        # input(examples)
        return target_score / len(sentid2ans), (distMean,distsd_,pixMean,pixsd_,scenarios,examples), tagging_score / len(sentid2ans), meta_score/ len(sentid2ans)
