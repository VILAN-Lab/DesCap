from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import json
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision.ops import boxes as box_ops


def apply_along_batch(func, M):
    #apply torch function for each image in a batch, and concatenate results back into a single tensor
    tensorList = [func(m) for m in torch.unbind(M, dim=0) ]
    result = torch.stack(tensorList, dim=0)
    return result

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        '''
        This function computes
            log(y_t) * reward * mask_t  (where mask_t zeroes out non-words in the sequence)
        given
            input = predicted probability
            sequence = predicted word index
            reward = ...
        '''

        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))


class NoamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0  # iter
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr':self.current_lr,
                'scheduler_state_dict': {key: value for key, value in self.scheduler.__dict__.items() if key not in {'optimizer', 'is_better'}},
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr) # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.__dict__.update(state_dict['scheduler_state_dict'])
            self.scheduler._init_is_better(mode=self.scheduler.mode, threshold=self.scheduler.threshold, threshold_mode=self.scheduler.threshold_mode)
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

def get_std_opt(optimizer, factor=1, warmup=2000):
    # return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    return NoamOpt(512, factor, warmup, optimizer)

def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding= True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    #returns a relational embedding for each pair of bboxes, with dimension = dim_g
    #follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    # f_g: torch.Size([32, 15, 4])
    batch_size = f_g.size(0)  # 32

    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)
    # x_min: torch.Size([32, 15, 1])

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    #cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)  # (32,15,15)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return(embedding)

def get_box_feats(boxes, d):
    """
    Given the bounding box coordinates for an object on an image, this function
    generates the trivial horizontal, and vertical 0-1 vector encoding of the bbox.

    This function is currently not used anywhere else in our codebase.
    """
    h,w = boxes.shape[:2]
    boxes_times_d = (d*boxes).astype(np.int32)
    boxes_wmin = boxes_times_d[:,:,0]
    boxes_wmax = boxes_times_d[:,:,2]
    boxes_hmin = boxes_times_d[:,:,1]
    boxes_hmax = boxes_times_d[:,:,3]

    box_hfeats = np.zeros((h,w,d))
    for i in range(h):
        for j in range(w):
            if not np.all(boxes_times_d[i,j]==np.zeros(4)):
                h_vector = np.concatenate([np.zeros(boxes_hmin[i,j]), np.ones(boxes_hmax[i,j]-boxes_hmin[i,j]), np.zeros(d-boxes_hmax[i,j])])
                box_hfeats[i,j]+=h_vector

    box_wfeats = np.zeros((h,w,d))
    for i in range(h):
        for j in range(w):
            if not np.all(boxes_times_d[i,j]==np.zeros(4)):
                w_vector = np.concatenate([np.zeros(boxes_wmin[i,j]), np.ones(boxes_wmax[i,j]-boxes_wmin[i,j]), np.zeros(d-boxes_wmax[i,j])])
                box_wfeats[i,j]+=w_vector
    return(box_hfeats, box_wfeats)

def single_image_get_box_feats(boxes, d):
    h = boxes.shape[0]
    boxes_times_d = (d*boxes).astype(np.int32)
    boxes_wmin = boxes_times_d[:,0]
    boxes_wmax = boxes_times_d[:,2]
    boxes_hmin = boxes_times_d[:,1]
    boxes_hmax = boxes_times_d[:,3]

    box_hfeats = np.zeros((h,d))
    for i in range(h):
        #for j in range(w):
            if not np.all(boxes_times_d[i]==np.zeros(4)):
                h_vector = np.concatenate([np.zeros(boxes_hmin[i]), np.ones(boxes_hmax[i]-boxes_hmin[i]), np.zeros(d-boxes_hmax[i])])
                box_hfeats[i]+=h_vector

    box_wfeats = np.zeros((h,d))
    for i in range(h):
        #for j in range(w):
            if not np.all(boxes_times_d[i]==np.zeros(4)):
                w_vector = np.concatenate([np.zeros(boxes_wmin[i]), np.ones(boxes_wmax[i]-boxes_wmin[i]), np.zeros(d-boxes_wmax[i])])
                box_wfeats[i]+=w_vector
    return(box_hfeats, box_wfeats)

def get_box_areas(arr):
    return((arr[:,2]-arr[:,0])*(arr[:,3]-arr[:,1]))

def torch_get_box_feats(boxes, d):
    device = boxes.device
    h,w = boxes.shape[:2]
    boxes_times_d = (d*boxes).type(torch.int32)
    boxes_wmin = boxes_times_d[:,:,0]
    boxes_wmax = boxes_times_d[:,:,2]
    boxes_hmin = boxes_times_d[:,:,1]
    boxes_hmax = boxes_times_d[:,:,3]

    box_hfeats = torch.zeros((h,w,d), device=device)
    zero_fourtuple=torch.zeros(4,dtype=torch.int32,device=device)

    for i in range(h):
        for j in range(w):
            if not torch.all(boxes_times_d[i,j]==zero_fourtuple):
                h_vector = torch.cat([torch.zeros(boxes_hmin[i,j], device=device), torch.ones(boxes_hmax[i,j]-boxes_hmin[i,j], device=device), torch.zeros(d-boxes_hmax[i,j], device=device)])
                box_hfeats[i,j]+=h_vector

    box_wfeats = torch.zeros((h,w,d), device=device)
    for i in range(h):
        for j in range(w):
            if not all(boxes_times_d[i,j]==zero_fourtuple):
                w_vector = torch.cat([torch.zeros(boxes_wmin[i,j], device=device), torch.ones(boxes_wmax[i,j]-boxes_wmin[i,j], device=device), torch.zeros(d-boxes_wmax[i,j], device=device)])
                box_wfeats[i,j]+=w_vector
    return(box_hfeats, box_wfeats)

def get_sub_sg(box, target):
    """

    Args:
        box: caption region box, dict
        target: image scene graph data, torch.Size([128, 4])

    Returns: caption box center sub scene graph data

    """
    # one item of batch_size
    device = torch.device("cpu")
    box = box.to(device)  # (128, 4)
    target = {k: v.to(device).numpy() for k, v in target.items()}
    sbj_box = target['sbj_box']     # (100, 4)
    sbj_feat = target['sbj_feat']   # (100, 2048)
    obj_box = target['obj_box']     # (100, 4)
    obj_feat = target['obj_feat']   # (100, 2048)
    rel_feat = target['rel_feat']   # (100, 1024)
    ############### find all object node ################
    node_box_list = []
    node_box_torch = []
    node_box_feat = []
    s0 = np.around(sbj_box[0]).astype('int')
    o0 = np.around(obj_box[0]).astype('int')
    node_box_list.append(s0.tolist())
    node_box_list.append(o0.tolist())
    node_box_torch.append(torch.from_numpy(sbj_box[0]).unsqueeze(0))
    node_box_torch.append(torch.from_numpy(obj_box[0]).unsqueeze(0))
    node_box_feat.append(torch.from_numpy(sbj_feat[0]).unsqueeze(0))
    node_box_feat.append(torch.from_numpy(obj_feat[0]).unsqueeze(0))

    na = {0: 's0', 1: 'o0'}
    for i in range(1, len(sbj_box)):
        tup = np.around(sbj_box[i]).astype('int')
        a = 0
        for rel in node_box_list:
            if rel != tup.tolist():
                a += 1
                if a == len(node_box_list):
                    node_box_list.append(tup.tolist())
                    node_box_torch.append(torch.from_numpy(sbj_box[i]).unsqueeze(0))
                    node_box_feat.append(torch.from_numpy(sbj_feat[i]).unsqueeze(0))
                    na[len(na)] = 's' + str(i)
                continue
            else:
                break
        tupe = np.around(obj_box[i]).astype('int')
        b = 0
        for rela in node_box_list:
            if rela != tupe.tolist():
                b += 1
                if b == len(node_box_list):
                    node_box_list.append(tupe.tolist())
                    node_box_torch.append(torch.from_numpy(obj_box[i]).unsqueeze(0))
                    node_box_feat.append(torch.from_numpy(obj_feat[i]).unsqueeze(0))
                    na[len(na)] = 'o' + str(i)
                continue
            else:
                break
    node_box_torch = torch.cat(node_box_torch, dim=0)
    node_box_feat = torch.cat(node_box_feat, dim=0)

    sbj_box_new = []
    obj_box_new = []
    for j in range(len(sbj_box)):
        s_box = np.around(sbj_box[j]).astype('int').tolist()
        o_box = np.around(obj_box[j]).astype('int').tolist()
        for on, object_node in enumerate(node_box_list):
            if s_box == object_node:
                sbj_box_new.append(na[on])
            if o_box == object_node:
                obj_box_new.append(na[on])
    sando = []
    for sa in range(len(sbj_box_new)):
        sando.append(sbj_box_new[sa] + obj_box_new[sa])

    ################ get adjacency matrix #################
    relation_matrix = [[0 for i in range(len(na))] for i in range(len(na))]  # 2d list
    relation_matrix_np = np.zeros((len(na), len(na)), dtype=int)  # np array
    relation_matrix_torch = torch.ones((len(na), len(na), 1024))  # torch.Size([41, 41, 1024])
    rso_index = []  # remind relation index in all 100
    for r in range(len(sbj_box)):  # 100
        x = np.around(sbj_box[r]).astype('int').tolist()
        y = np.around(obj_box[r]).astype('int').tolist()
        z2 = torch.from_numpy(rel_feat[r])
        z = 1
        for ir, rr in enumerate(node_box_list):
            if x == rr:
                x_index = ir
            if y == rr:
                y_index = ir
        if relation_matrix[x_index][y_index] == 0:
            rso_index.append(r)
        relation_matrix[x_index][y_index] = int(z)
        relation_matrix_np[x_index, y_index] = int(z)
        relation_matrix_torch[x_index, y_index, :] = z2

    ################ find sub scene graph #################
    # print(box[0])  # caption region box
    iou = box_ops.box_iou(box, node_box_torch)
    ma, ma_idx = torch.max(iou, dim=1)
    sub_sg_matrix_all = []
    #sub_sg_rel_feat_all = []
    #sub_sg_matrix_01_all = []
    sub_sg_node_box_all = []
    sub_sg_node_feat_all = []
    for ma_idx_i in ma_idx.numpy():
        ss = relation_matrix_np[ma_idx_i, :]  # 8 row
        oo = relation_matrix_np[:, ma_idx_i]  # 8 col

        huo = np.asarray([ss[i]|oo[i] for i in range(len(ss))])  # node, where is 1
        sub_sg_node_idx = torch.nonzero(torch.from_numpy(huo)).squeeze(-1).tolist()
        sub_sg_node_idx.insert(0, ma_idx_i)  # # self node
        sub_sg_node = [na[l] for l in sub_sg_node_idx]
        sub_sg_matrix = torch.ones((len(sub_sg_node_idx), len(sub_sg_node_idx), 1024))
        #sub_sg_matrix_01 = np.zeros((len(sub_sg_node_idx),len(sub_sg_node_idx)),dtype=int)
        #sub_sg_rel_feat = []
        for si, sub_node_i in enumerate(sub_sg_node_idx):
            for sj, sub_node_j in enumerate(sub_sg_node_idx):
                sub_sg_matrix[si, sj, :] = relation_matrix_torch[sub_node_i, sub_node_j]
                #aaa = relation_matrix_torch[sub_node_i, sub_node_j]
                #if all(aaa[:] == 1):
                    #sub_sg_matrix_01[si,sj] = 0
                #else:
                    #sub_sg_matrix_01[si,sj] = 1
                    #sub_sg_rel_feat.append(relation_matrix_torch[sub_node_i, sub_node_j].unsqueeze(0))
        
        #sub_sg_rel_feat = torch.cat(sub_sg_rel_feat, dim=0)

        sub_sg_node_feat = node_box_feat[sub_sg_node_idx]
        sub_sg_node_box = node_box_torch[sub_sg_node_idx]
        #sub_sg_rel_feat_all.append(sub_sg_rel_feat)
        #sub_sg_matrix_01_all.append(sub_sg_matrix_01)
        sub_sg_matrix_all.append(sub_sg_matrix.cuda())
        sub_sg_node_box_all.append(sub_sg_node_box.cuda())
        sub_sg_node_feat_all.append(sub_sg_node_feat.cuda())
        #print(sub_sg_matrix.shape,sub_sg_node_box.shape,sub_sg_node_feat.shape)

    return sub_sg_matrix_all, sub_sg_node_box_all, sub_sg_node_feat_all

def get_sub_sg2(region_feat, region_box, sg_node_box, sg_node_feat, sg_index):
    """

    Args:
        box: caption region box, dict
        target: image scene graph data, torch.Size([128, 4])

    Returns: caption box center sub scene graph data

    """
    ious = box_ops.box_iou(region_box, sg_node_box)
    _, max_ind = torch.max(ious, dim=1)
    all_subg_node_feat = []
    all_subg_node_box = []
    #iou_loss = []
    for j, i in enumerate(max_ind):
        ii = [i.tolist()]
        sbj_idx = torch.nonzero(sg_index[0]==i).squeeze()
        obj_idx = torch.nonzero(sg_index[1]==i).squeeze()
        subg_s = sg_index[1][sbj_idx].tolist()
        subg_o = sg_index[0][obj_idx].tolist()
        if type(subg_s) != list:
            subg_s = [subg_s]
        if type(subg_o) != list:
            subg_o = [subg_o]

        subg_node_idx = list(set(subg_s + subg_o + ii))
        subg_node_feat = sg_node_feat[subg_node_idx]
        subg_node_box = sg_node_box[subg_node_idx]

        region_feature = region_feat[j].unsqueeze(0)
        region_bbox = region_box[j].unsqueeze(0)

        subg_node_feat = torch.cat((subg_node_feat, region_feature), dim=0)
        subg_node_box = torch.cat((subg_node_box, region_bbox), dim=0)
        #iou_loss.append(torch.sum(box_ops.box_iou(region_bbox, subg_node_box), dim=-1))

        all_subg_node_feat.append(subg_node_feat)
        all_subg_node_box.append(subg_node_box)
    #iou_loss = torch.mean(torch.cat((iou_loss), dim=-1))
    #return all_subg_node_feat, all_subg_node_box, iou_loss
    return all_subg_node_feat, all_subg_node_box

def generate_sg_matrix(s, r, o):
    # s, r, o is label
    relation_list = []
    for i in range(len(s)):
        tup = (str(int(s[i])), str(int(o[i])), str(int(r[i])))
        relation_list.append(tup)
    member_dict = {}
    member_index = 0
    for name_tuple in relation_list:
        for name in name_tuple[:2]:
            if name in member_dict:
                continue
            member_dict[name] = member_index
            member_index += 1
    member_dict_re = dict(zip(member_dict.values(), member_dict.keys()))
    relation_matrix = [[0 for i in range(len(member_dict))] for i in range(len(member_dict))]
    for (x, y, z) in relation_list:
        x_index = member_dict[x]
        y_index = member_dict[y]
        relation_matrix[x_index][y_index] = int(z)

    rr = []
    for co in range(len(relation_matrix)):
        for ro in range(len(relation_matrix)):
            if relation_matrix[co][ro] == 0:
                continue
            tri = (member_dict_re[co], member_dict_re[ro], str(relation_matrix[co][ro]))
            rr.append(tri)
    r_index = []
    for ri in rr:
        for rj, rk in enumerate(relation_list):
            if operator.eq(ri, rk):
                r_index.append(rj)
                break
    rr_list = []
    for (rs,ro,rp) in rr:
        a = [int(rs),int(ro),int(rp)]
        rr_list.append(a)

    return relation_matrix, rr_list, r_index

def generate_sg_matrix_box(box_gt, targets):
    # s, o is box , r is label
    device = torch.device("cpu")
    box_gt = box_gt[0].to(device)  # (128, 4)
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
    targets = targets[0]
    sbj_box = targets['sbj_box']
    sbj_feat = targets['sbj_feat']
    obj_box = targets['obj_box']
    obj_feat = targets['obj_feat']
    rel_feat = targets['rel_feat']
    relation_list = []
    s0 = np.around(np.array([sbj_box[0][0], sbj_box[0][1], sbj_box[0][2], sbj_box[0][3]])).astype('int')
    o0 = np.around(np.array([obj_box[0][0], obj_box[0][1], obj_box[0][2], obj_box[0][3]])).astype('int')
    relation_list.append(s0.tolist())
    relation_list.append(o0.tolist())
    sbj_idx = [0]
    obj_idx = [0]
    na = {0:'s0', 1:'o0'}
    for i in range(1, len(sbj_box)):
        tup = np.around(np.array([sbj_box[i][0], sbj_box[i][1], sbj_box[i][2], sbj_box[i][3]])).astype('int')
        a = 0
        for rel in relation_list:
            if rel != tup.tolist():
                a += 1
                if a == len(relation_list):
                    relation_list.append(tup.tolist())
                    sbj_idx.append(i)
                    na[len(na)] = 's' + str(i)
                continue
            else:
                break
        tupe = np.around(np.array([obj_box[i][0], obj_box[i][1], obj_box[i][2], obj_box[i][3]])).astype('int')
        b = 0
        for rela in relation_list:
            if rela != tupe.tolist():
                b += 1
                if b == len(relation_list):
                    relation_list.append(tupe.tolist())
                    obj_idx.append(i)
                    na[len(na)] = 'o' + str(i)
                continue
            else:
                break
    node_num = len(relation_list)
    na_list = sbj_idx + obj_idx
    rel_idx = set(na_list)
    node_box = []
    #print(sbj_idx)
    #print(obj_idx)
    #print(na)
    #print(node_num)
    for k,v in na.items():
        if v[0] == 's':
            boxs = sbj_box[int(v[1:])]
        elif v[0] == 'o':
            boxs = obj_box[int(v[1:])]
        else:
            print('no such key')
        node_box.append(boxs)

    relation_matrix = [[0 for i in range(len(na))] for i in range(len(na))]
    rso_index = []
    sbj_idx_all = []
    obj_idx_all = []
    for r in range(len(sbj_box)):  # 100
        x = np.around(np.array([sbj_box[r][0], sbj_box[r][1], sbj_box[r][2], sbj_box[r][3]])).astype('int').tolist()
        y = np.around(np.array([obj_box[r][0], obj_box[r][1], obj_box[r][2], obj_box[r][3]])).astype('int').tolist()
        #z = rel_label[r]
        z = 1
        for ir,rr in enumerate(node_box):
            rr = np.around(np.array([rr[0], rr[1], rr[2], rr[3]])).astype('int').tolist()
            if x == rr:
                x_index = ir
            if y == rr:
                y_index = ir
        if relation_matrix[x_index][y_index] == 0:
            rso_index.append(r)
            sbj_idx_all.append(na[x_index])
            obj_idx_all.append(na[y_index])
        relation_matrix[x_index][y_index] = int(z)
    sg_box_s = []
    for g in sbj_idx_all:
        if g[0] == 's':
            sg_box_s.append(sbj_box[int(g[1:])].unsqueeze(0))
        else:
            sg_box_s.append(obj_box[int(g[1:])].unsqueeze(0))
    sg_box_o = []
    for h in obj_idx_all:
        if h[0] == 's':
            sg_box_o.append(sbj_box[int(h[1:])].unsqueeze(0))
        else:
            sg_box_o.append(obj_box[int(h[1:])].unsqueeze(0))

    sg_box_s = torch.cat(sg_box_s, 0)
    sg_box_o = torch.cat(sg_box_o, 0)
    node_box = [i.unsqueeze(0) for i in node_box]
    node_box = torch.cat(node_box, 0)
    iou = box_ops.box_iou(box_gt, node_box)
    ma, ma_idx = torch.max(iou, dim=1)

    sub_sbj_idx_all = []
    sub_obj_idx_all = []
    #sub_sbj_box_all = []
    #sub_obj_box_all = []
    for kk in ma_idx.tolist():
        sub_sbj_idx = []
        sub_obj_idx = []
        #sub_sbj_box = []
        #sub_obj_box = []
        for vv,q in enumerate(sbj_idx_all):
            if na[kk] == q:
                sub_sbj_idx.append(vv)
                #sub_obj_box.append(q)
        sub_sbj_idx_all.append(sub_sbj_idx)
        #sub_obj_box_all.append(sub_obj_box)

        for vvv,qq in enumerate(obj_idx_all):
            if na[kk] == qq:
                sub_obj_idx.append(vvv)
                #sub_sbj_box.append(qq)
        sub_obj_idx_all.append(sub_obj_idx)
        #sub_sbj_box_all.append(sub_sbj_box)
    #print(sub_sbj_idx_all[0],sub_obj_idx_all[0])
    #print(sub_sbj_box_all[1],sub_obj_box_all[1])
    for mn in range(len(ma_idx.tolist())):
        l1 = sub_sbj_idx_all[mn]
        l2 = sub_obj_idx_all[mn]
        l3 = l1 + l2
        l3 = set(l3)
        assert len(l1)+len(l2) == len(l3)
    # get sub sg rel feat
    # get object node
    print('all rel num:',len(rso_index),',all object num:', len(na))
    relation_feature = []  # sub sg rel feat
    sg_node_box = []  # sub sg node box
    sg_node_feat = []  # sub sg node feat
    for nm in range(len(ma_idx.tolist())):
        aa1 = []
        bb1 = []
        node_object_boxes = []
        node_object_feats = []
        all_index = sub_sbj_idx_all[nm]+sub_obj_idx_all[nm]
        relation_feature.append(rel_feat[rso_index][all_index])
        #print(rel_feat[rso_index][all_index].shape)
        center = sbj_idx_all[sub_sbj_idx_all[nm][0]]
        for aa in sub_sbj_idx_all[nm]:
            aa1.append(obj_idx_all[aa])
        for bb in sub_obj_idx_all[nm]:
            bb1.append(sbj_idx_all[bb])
        st = list(set(aa1+bb1))
        st.insert(0, center)
        for node_object_box in st:
            if node_object_box[0] == 's':
                node_object_boxes.append(sbj_box[int(node_object_box[1:])].unsqueeze(0))
                node_object_feats.append(sbj_feat[int(node_object_box[1:])].unsqueeze(0))
            else:
                node_object_boxes.append(obj_box[int(node_object_box[1:])].unsqueeze(0))
                node_object_feats.append(obj_feat[int(node_object_box[1:])].unsqueeze(0))

        node_object_boxes = torch.cat(node_object_boxes, 0)
        node_object_feats = torch.cat(node_object_feats, 0)
        #print(node_object_boxes.shape)
        #print(node_object_feats.shape)
        sg_node_box.append(node_object_boxes)
        sg_node_feat.append(node_object_feats)

    #return relation_matrix, sbj_idx_all, obj_idx_all, rso_index
    return relation_matrix, relation_feature, sg_node_box, sg_node_feat


    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion2(nn.Module):  # NLLLoss
    def __init__(self):
        super(LanguageModelCriterion2, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        #input: (batch_size, seq_len, vocab_size)
        input = input.contiguous().view(-1, input.size(2))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        print(input.size(), target.size(), mask.size())

        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def want_to_continue(found_issue):
    print('--' * 10)
    print(found_issue + '. Would you like to continue? [y/N]')

    yes = {'yes','y', 'ye', 'Y'}
    no = {'no','n','','N'}

    choice = raw_input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'y' or 'N'")
