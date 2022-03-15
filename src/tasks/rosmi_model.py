# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 25

MAX_BOXES = 73

class ROSMIModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        self.hid_dim = self.lxrt_encoder.dim

        self.distance_fc = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim*6),
            # GeLU(),
            GeLU(),
            BertLayerNorm(self.hid_dim*6, eps=1e-12),
            nn.Linear(self.hid_dim*6, 2)
        )

        self.landmark_feed = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim*6),
            # GeLU(),
            GeLU(),
            BertLayerNorm(self.hid_dim*6, eps=1e-12),
            nn.Linear(self.hid_dim*6, 2)
        )
        # num_bearings = 9
        self.bearing_fc = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim*6),
            # GeLU(),
            GeLU(),
            BertLayerNorm(self.hid_dim*6, eps=1e-12),
            nn.Linear(self.hid_dim*6, 9)
        )
        self.land_cl = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim*7),
            # GeLU(),
            GeLU(),
            BertLayerNorm(self.hid_dim*7, eps=1e-12),
            nn.Linear(self.hid_dim*7, 1)
        )
        # OLD
        self.land_fc = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim*4),
            GeLU(),
            BertLayerNorm(self.hid_dim*4, eps=1e-12),
            nn.Linear(self.hid_dim*4, 4)
        )
        # OLD
        self.logit_fc = nn.Sequential(
            # nn.Linear(68 * self.hid_dim* 3, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim*4),
            GeLU(),
            BertLayerNorm(self.hid_dim*4, eps=1e-12),
            nn.Linear(self.hid_dim*4, 4)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.land_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.land_cl.apply(self.lxrt_encoder.model.init_bert_weights)
        self.bearing_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.distance_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.landmark_feed.apply(self.lxrt_encoder.model.init_bert_weights)


    def forward(self, feat, feat_mask, pos, names, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param names:  (b, o, max_seq_length)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        feat_seq = self.lxrt_encoder(sent, (feat, pos, names),visual_attention_mask = feat_mask)

        # 0 = language 1 = vision
        # print(feat_seq[0].shape)
        # input(feat_seq[1].shape)
        # if args.n_ent:
        #     x = self.lxrt_encoder(sent, (feat, pos, names),visual_attention_mask = feat_mask)
        # else:
        #     x = self.lxrt_encoder(sent, (feat, pos, names))
        # # print(x)
        # print((x.shape))
        # input(torch.mean(x))
        # x = x.view(-1, 68 * self.hid_dim* 3)
        # print(x.shape)
        logit = self.logit_fc(feat_seq[1][:,0])
        dist = self.distance_fc(feat_seq[0])
        land_uni = self.landmark_feed(feat_seq[0])
        # dist_e = self.distance_fc(feat_seq[0])


        dist_s, dist_e = dist.split(1, dim=-1)
        # print(dist_s.shape)
        dist_s = dist_s.squeeze(-1)
        dist_e = dist_e.squeeze(-1)


        land_uni_s, land_uni_e = land_uni.split(1, dim=-1)
        # print(dist_s.shape)
        start_logits = land_uni_s.squeeze(-1)
        end_logits = land_uni_e.squeeze(-1)


        landmark_ = self.land_fc(feat_seq[1][:,0])

        cland_ = self.land_cl(feat_seq[1])
        cland_ = cland_.squeeze(-1)
        bearing_ = self.bearing_fc(feat_seq[0][:,0])
        return logit, (dist_s,dist_e, landmark_,cland_, bearing_,start_logits,end_logits)
