from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .graph_conv import _GraphConvolutionLayer
from utils.box_utils import get_sub_sg2


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

"""
GCN backbone: integrate context information within the graph and update node and edge features
"""
class GCN(nn.Module):
    def __init__(self, GCN_layers=2, GCN_dim=1024, GCN_residual=2, GCN_use_bn=False):
        super(GCN, self).__init__()
        self.GCN_layers = GCN_layers
        self.GCN_dim = GCN_dim
        self.GCN_residual = GCN_residual
        self.GCN_use_bn = GCN_use_bn
        if self.GCN_layers != 0:
            self.make_graph_encoder()

        self.node_change = nn.Linear(256, 1024, bias=True)
        self.edge_change = nn.Linear(1024, 1024, bias=True)
        normal_init(self.node_change, 0, 0.001)
        normal_init(self.edge_change, 0, 0.001)

        self.wq = nn.Linear(256, 1024)
        self.wk = nn.Linear(1024, 1024)
        self.wv = nn.Linear(1024, 1024)
        nn.init.kaiming_uniform_(self.wq.weight)
        nn.init.constant_(self.wq.bias, 0)
        nn.init.kaiming_uniform_(self.wk.weight)
        nn.init.constant_(self.wk.bias, 0)
        nn.init.kaiming_uniform_(self.wv.weight)
        nn.init.constant_(self.wv.bias, 0)

        self.dropout = nn.Dropout(0.1)

    def make_graph_encoder(self):
        self.gcn = nn.ModuleList()
        for i in range(self.GCN_layers):
            self.gcn.append(_GraphConvolutionLayer(1024, 1024, self.GCN_dim, use_bn=self.GCN_use_bn))

    def forward(self, region_feat, region_box, target_sg):
        #b, N, K, L, obj_feats, rel_feat, rel_ind = 0
        ori_x_obj = target_sg[0]['sg_node_feat'].unsqueeze(0) # object feature vectors
        ori_x_pred = target_sg[0]['sg_edge_feat'].unsqueeze(0) # predicate feature vectors
        rel_ind = target_sg[0]['sg_object_ind'].unsqueeze(0)
        sg_object_box = target_sg[0]['sg_node_box']

        b = 1
        N, node_dim = ori_x_obj.shape[1], ori_x_obj.shape[2]
        K, rel_dim = ori_x_pred.shape[1], ori_x_pred.shape[2]
        L = self.GCN_dim

        # ----------change dim----------------#
        x_obj1 = self.node_change(ori_x_obj)
        #x_pred1 = self.edge_change(ori_x_pred)
        x_pred1 = ori_x_pred

        if self.GCN_layers != 0:
            attend_score = x_obj1.data.new(b,K).fill_(1)

            # ajacency map in GCN
            if rel_ind.dtype != torch.int64:
                rel_ind = rel_ind.long()
            map_obj_rel = self.make_map(x_obj1, attend_score, b, N, K, rel_ind[:,0,:], rel_ind[:,1,:])

            # GCN feed forward
            for i, gcn_layer in enumerate(self.gcn):
                x_obj, x_pred = gcn_layer(x_obj1, x_pred1, map_obj_rel)
                # residual skip connection
                if (i+1) % self.GCN_residual == 0:
                    x_obj = x_obj + x_obj1
                    x_obj1 = x_obj
                    x_pred = x_pred + x_pred1
                    x_pred1 = x_pred

        region_feat = self.wq(region_feat)  # 2048 --> 1024
        # ------------- locat subgraph--------------#
        # subg_feat, subg_box = get_sub_sg2(region_feat, region_box, sg_object_box, x_obj1[0], rel_ind[0])

        # # -------- pad 0 to max_num-----------------#
        # # 一个区域一个子图,pad成 (64, max_num, 1024),用于Transformer
        # subg_node_num = [i.shape[0] for i in subg_feat]
        # pad_subg_feat = torch.zeros([len(subg_node_num),max(subg_node_num),1024], dtype=torch.float32).cuda()
        # pad_subg_box = torch.zeros([len(subg_node_num),max(subg_node_num),4], dtype=torch.float32).cuda()
        # for i, j in enumerate(subg_node_num):
        #     pad_subg_feat[i,:j,:] = subg_feat[i]
        #     pad_subg_box[i,:j,:] = subg_box[i]

        # ---------- region guided attention (fusion)-----------#
        # 一个区域一个1024D向量,一般用于LSTM
        #region_feature = self.attention(region_feat, subg_feat)

        # ablation for sub_graph_extraction
        n = sg_object_box.shape[0] + 1
        subg_node_num = [n for i in region_feat]
        pad_subg_feat = torch.zeros([len(subg_node_num),max(subg_node_num),1024], dtype=torch.float32).cuda()
        pad_subg_box = torch.zeros([len(subg_node_num),max(subg_node_num),4], dtype=torch.float32).cuda()
        for i, j in enumerate(subg_node_num):
            pad_subg_feat[i,:1,:] = region_feat[i]
            pad_subg_feat[i,1:,:] = x_obj1[0]
            pad_subg_box[i,:1,:] = region_box[i]
            pad_subg_box[i,1:,:] = sg_object_box

        return pad_subg_feat, pad_subg_box, subg_node_num

    def make_map(self, x_obj, attend_score, batch_size, N, K, ind_subject, ind_object):
        """
        generate GCN mapping between subject and predicate, and between object and predicate
        """

        # map between sub object and obj object
        map_sobj_rel = x_obj.data.new(batch_size, N, K).zero_()
        map_oobj_rel = x_obj.data.new(batch_size, N, K).zero_()
        for i in range(batch_size):
            map_sobj_rel[i].scatter_(0, ind_subject[i].contiguous().view(1, K), attend_score[i].contiguous().view(1,K)) # row is target, col is source
            map_oobj_rel[i].scatter_(0, ind_object[i].contiguous().view(1, K), attend_score[i].contiguous().view(1,K)) # row is target, col is source
        map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 3)  # [b, N, K, 2]

        return map_obj_rel

    def attention(self, guided, inp, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        # guided-attention
        #query_all = self.wq(guided)
        query_all = guided
        d_k = query_all.size(-1)
        all_feat = []
        for i in inp:
            query = query_all[0].unsqueeze(0)
            key = self.wk(i)
            value = self.wv(i)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim = -1)
            p_attn = self.dropout(p_attn)
            att_out = torch.matmul(p_attn, value)
            all_feat.append(att_out)
        all_region_feat = torch.cat(all_feat, dim=0)
        return all_region_feat













