import h5py
import pickle
import json
import os
import cv2
import operator
import numpy as np
#from dataset import DenseCapDataset
#from utils.box_utils import generate_sg_matrix
from tqdm import tqdm


feat_data = h5py.File('/media/ubuntu/HYJ/dataset/mvsa_4511_image_visual_feat.h5', 'r')
sg_data = h5py.File('/home/ubuntu/D/hyj/project/msa/data/mvsa_4511_sg64.h5', 'r')
densecap_split = np.load('/home/ubuntu/D/hyj/project/msa/data/for_msa_1.npz')['arr_0']

p1 = 'data/data_new/VG-regions-dicts-lite-v1.0-new.pkl'
p2 = 'data/data_new/VG-regions-lite-v1.0-new.h5'
p3 = 'data/feat_data/val.hdf5'
p4 = 'data/feat_data/val_imgid2idx.pkl'
p5 = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/densecap-sg100/sg100-val.h5'
p6 = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/densecap-sg100/sg100-train-all-box.h5'
p7 = '/home/ubuntu/D/hyj/data/coco-caption-data/coco_img_sg/9.npy'
p8 = '/home/ubuntu/D/hyj/data/coco-caption-data/spice_sg_dict2.npz'
p9 = '/home/ubuntu/D/hyj/data/coco-caption-data/rela_dict.npy'
p10 = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/region_descriptions_v1.0.json'
p11 = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/image_data_v1.0.json'
p12 = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K/1168.jpg'
p12 = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K_2/1168.jpg'

IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
VG_DATA_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-dicts-lite.pkl'
dcd = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='val')
img, targets, info = dcd.__getitem__(0)

re10 = json.load(open(p10, 'r'))
da10 = json.load(open(p11, 'r'))
pkl = pickle.load(open(p1, 'rb'))

h5 = h5py.File(p2, 'r')

for i in h5.keys():
    print(i)
    print(h5[i][:].shape)
box1 = h5['boxes'][0]
cap1 = h5['captions'][0]


all_box = h5['boxes'][:]
l = all_box.shape[0]
for j in tqdm(range(l)):
    boxes = np.asarray(all_box[j])
    for k in range(len(boxes)):
        if boxes[k] < 0:
            print(j,boxes)

i2w1 = np.load(p8, allow_pickle=True)['spice_dict'][()]['ix_to_word']
i2w2 = np.load(p9, allow_pickle=True)[()]
da = np.load(p7, allow_pickle=True, encoding="latin1")[()]

for i,j in da.items():
    print(i, j.shape)
print(da['rela_matrix'])
print(da['obj_attr'][0])

object_name = json.load(open('/home/ubuntu/D/hyj/project/Large-Scale-VRD.pytorch-master/data/vg/objects.json','r'))
predict_name = json.load(open('/home/ubuntu/D/hyj/project/Large-Scale-VRD.pytorch-master/data/vg/predicates.json','r'))


pa = os.path.join(IMG_DIR_ROOT, info['dir'],info['file_name'])
sg_sbj_box = targets['sg_sbj_box']
sg_obj_box = targets['sg_obj_box']
rcnn_sbj_box = targets['rcnn_sbj_box']
rcnn_obj_box = targets['rcnn_obj_box']
# densecap_val = json.load(open('info/densecap_splits.json', 'r'))['val']
# h5_sg = h5py.File(p5, 'r')
# pkl = pickle.load(open(p4, 'rb'))
# idx2imgid = dict(zip(pkl.values(), pkl.keys()))
# h5 = h5py.File(p3, 'r')
# for i in h5_sg.keys():
#     print(i)
#     print(h5_sg[i][:].shape)
# idx = 500
# sg_idx = densecap_val.index(idx2imgid[idx])
# id = str(idx2imgid[idx]) + '.jpg'
# box_num = int(h5['num_boxes'][idx])
# box = h5['image_bb'][idx]
#
# # sg box
# sg_obj_box = h5_sg['det_boxes_o_top'][sg_idx]
# sg_sbj_box = h5_sg['det_boxes_s_top'][sg_idx]
#
# if os.path.exists('/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K/' + id):
#     pa = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K/' + id
# else:
#     pa = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K_2/' + id
#
im = cv2.imread(pa)
src = np.array(im)
src2 = np.array(im)
# faster rcnn
for i in range(len(rcnn_obj_box)):
    topleft00 = np.around(np.array([rcnn_obj_box[i][0], rcnn_obj_box[i][1]])).astype('int')
    bottomright00 = np.around(np.array([rcnn_obj_box[i][2], rcnn_obj_box[i][3]])).astype('int')
    topleft11 = np.around(np.array([rcnn_sbj_box[i][0], rcnn_sbj_box[i][1]])).astype('int')
    bottomright11 = np.around(np.array([rcnn_sbj_box[i][2], rcnn_sbj_box[i][3]])).astype('int')
    point_color00 = (0, 255, 0)
    point_color11 = (255, 0, 0)
    thickness = 2
    lineType = 4
    cv2.rectangle(src, tuple(topleft00), tuple(bottomright00), point_color00, thickness, lineType)
    cv2.rectangle(src, tuple(topleft11), tuple(bottomright11), point_color11, thickness, lineType)
cv2.namedWindow('rcnn', cv2.WINDOW_AUTOSIZE)
cv2.imshow('rcnn', src)

# sg100
for i in range(len(sg_obj_box)):
    topleft0 = np.around(np.array([sg_obj_box[i][0], sg_obj_box[i][1]])).astype('int')
    bottomright0 = np.around(np.array([sg_obj_box[i][2], sg_obj_box[i][3]])).astype('int')
    topleft1 = np.around(np.array([sg_sbj_box[i][0], sg_sbj_box[i][1]])).astype('int')
    bottomright1 = np.around(np.array([sg_sbj_box[i][2], sg_sbj_box[i][3]])).astype('int')
    point_color0 = (0, 255, 0)
    point_color1 = (255, 0, 0)
    thickness = 2
    lineType = 4
    cv2.rectangle(src2, tuple(topleft0), tuple(bottomright0), point_color0, thickness, lineType)
    cv2.rectangle(src2, tuple(topleft1), tuple(bottomright1), point_color1, thickness, lineType)

cv2.namedWindow('sg', cv2.WINDOW_AUTOSIZE)
cv2.imshow('sg', src2)

cv2.waitKey(0)
cv2.destroyAllWindows()

h5 = h5py.File(p5, 'r')
vocab = pkl['idx_to_token']
f = pkl['idx_to_filename'][0]
id = int(f[:-4])
for i in h5.keys():
    print(i)
    print(h5[i][:].shape)
num_rel = int(h5['det_rel_num'][2])
l_s = h5['det_labels_s_top'][2][:num_rel]
rel = h5['det_labels_p_top'][2][:num_rel]
o_s = h5['det_labels_o_top'][2][:num_rel]
out = generate_sg_matrix(l_s,rel,o_s)
relation_list = []
for i in range(num_rel):
    tup = (str(int(l_s[i])), str(int(o_s[i])), str(int(rel[i])))
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
for (x,y,z) in relation_list:
    x_index = member_dict[x]
    y_index = member_dict[y]
    relation_matrix[x_index][y_index] = int(z)
relation_matrix_1d = [i for it in relation_matrix for i in it]

r = []
for co in range(len(relation_matrix)):
    for ro in range(len(relation_matrix)):
        if relation_matrix[co][ro] == 0:
            continue
        tri = (member_dict_re[co], member_dict_re[ro], str(relation_matrix[co][ro]))
        r.append(tri)
r_index = []
for ri in r:
    for rj, rk in enumerate(relation_list):
        if operator.eq(ri, rk):
            r_index.append(rj)
            break
r_after = []
for ide in r_index:
    r_after.append(relation_list[ide])
for x in range(100):
    l_ss = h5['det_labels_s_top'][2][x]
    ree = h5['det_labels_p_top'][2][x]
    o_ss = h5['det_labels_o_top'][2][x]
    print(object_name[int(l_ss)],'-',predict_name[int(ree)],'-',object_name[int(o_ss)])

cap_token = h5['captions'][1000]
print(cap_token)
# [  1  41   8 330  42   2   0   0   0   0   0   0   0   0   0   0   0]  # len = 17 (after add <bos> and <eos>)
cap_str = [' '.join(vocab[idx] for idx in cap_token.tolist())]
print(cap_str)
# ['<bos> window is very large <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>']  # len = 17
a = 1
