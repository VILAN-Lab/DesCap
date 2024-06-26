import h5py
import pickle
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torchvision.ops import boxes as box_ops

p1 = '../data/VG-regions-dicts-lite.pkl'
p2 = '../data/VG-regions-lite.h5'
p3 = '../data/feat_data/val.hdf5'
p4 = '../data/feat_data/val_imgid2idx.pkl'
p5 = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/densecap-sg100/sg100-val.h5'

train_data_file = '../data/feat_data_new/train.hdf5'
val_data_file = '../data/feat_data_new/val.hdf5'
test_data_file = '../data/feat_data_new/test.hdf5'

densecap_val = json.load(open('../info/densecap_splits.json', 'r'))['val']
h5_sg = h5py.File(p5, 'r')
pkl = pickle.load(open(p4, 'rb'))
idx2imgid = dict(zip(pkl.values(), pkl.keys()))
h5 = h5py.File(p3, 'r')

#h_train = h5py.File(train_data_file, "w")
h_val = h5py.File(val_data_file, "w")
#h_test = h5py.File(test_data_file, "w")
#train_img_features = h_train.create_dataset('image_features', (77398, 100, 2048), 'f')
#train_img_bb = h_train.create_dataset('image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
#train_num_boxes = h_train.create_dataset('num_boxes', (len(train_imgids),), 'f')

val_obj_features = h_val.create_dataset('obj_features', (5000, 100, 2048), 'f')
val_sbj_features = h_val.create_dataset('sbj_features', (5000, 100, 2048), 'f')
val_num_rel = h_val.create_dataset('num_rel', (5000,), 'f')

test_img_features = h_test.create_dataset('obj_features', (5000, 100, 2048), 'f')
test_num_rel = h_test.create_dataset('num_rel', (5000,), 'f')

for i, j in enumerate(tqdm(densecap_val)):
    id = str(j) + '.jpg'
    idx = pkl[j]
    box_num = int(h5['num_boxes'][idx])
    box = torch.from_numpy(h5['image_bb'][idx][:box_num])
    img_feat = torch.from_numpy(h5['image_features'][idx][:box_num])
    # sg box
    sg_obj_box = torch.from_numpy(h5_sg['det_boxes_o_top'][i])
    sg_sbj_box = torch.from_numpy(h5_sg['det_boxes_s_top'][i])
    # iou --> get max index
    iou_obj = box_ops.box_iou(sg_obj_box, box)
    iou_sbj = box_ops.box_iou(sg_sbj_box, box)
    max1, max_idx1 = torch.max(iou_obj, dim=1)
    max2, max_idx2 = torch.max(iou_sbj, dim=1)
    # get new image feature
    o_img_feat = img_feat[max_idx1]
    s_img_feat = img_feat[max_idx2]
    a = 1

# idx = 500
# id = str(idx2imgid[idx]) + '.jpg'
# box_num = int(h5['num_boxes'][idx])
# box = torch.from_numpy(h5['image_bb'][idx][:box_num])
#
# if os.path.exists('/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K/' + id):
#     pa = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K/' + id
# else:
#     pa = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K_2/' + id
#
# im = cv2.imread(pa)
# src = np.array(im)
# src2 = np.array(im)
#
# # sg box
# sg_idx = densecap_val.index(idx2imgid[idx])
# sg_obj_box = torch.from_numpy(h5_sg['det_boxes_o_top'][sg_idx])
# sg_sbj_box = torch.from_numpy(h5_sg['det_boxes_s_top'][sg_idx])
#
# iou_obj = box_ops.box_iou(sg_obj_box, box)
# iou_sbj = box_ops.box_iou(sg_sbj_box, box)
# max1, max_idx1 = torch.max(iou_obj, dim=1)
# max2, max_idx2 = torch.max(iou_sbj, dim=1)


# # faster rcnn
# box_new = box[max_idx2]
# for i in range(5):
#     topleft0 = np.around(np.array([box_new[i][0], box_new[i][1]])).astype('int')
#     bottomright0 = np.around(np.array([box_new[i][2], box_new[i][3]])).astype('int')
#     point_color0 = (0, 255, 0)
#     thickness = 2
#     lineType = 4
#     cv2.rectangle(src, tuple(topleft0), tuple(bottomright0), point_color0, thickness, lineType)
# cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('image', src)
#
# # sg100
# for i in range(5):
#     topleft0 = np.around(np.array([sg_obj_box[i][0], sg_obj_box[i][1]])).astype('int')
#     bottomright0 = np.around(np.array([sg_obj_box[i][2], sg_obj_box[i][3]])).astype('int')
#     topleft1 = np.around(np.array([sg_sbj_box[i][0], sg_sbj_box[i][1]])).astype('int')
#     bottomright1 = np.around(np.array([sg_sbj_box[i][2], sg_sbj_box[i][3]])).astype('int')
#     point_color0 = (0, 255, 0)
#     point_color1 = (255, 0, 0)
#     thickness = 2
#     lineType = 4
#     #cv2.rectangle(src2, tuple(topleft0), tuple(bottomright0), point_color0, thickness, lineType)
#     cv2.rectangle(src2, tuple(topleft1), tuple(bottomright1), point_color1, thickness, lineType)
#
# cv2.namedWindow('image_sg', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('image_sg', src2)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

a = 1