import pickle
import os
import h5py
import json



feat_data =  h5py.File('/media/ubuntu/disk1/hyj/denC/DenseCaption/train.hdf5', 'r')
imgid2idx = pickle.load(open('/media/ubuntu/disk1/hyj/denC/DenseCaption/train_imgid2idx.pkl', 'rb'))
densecap_split = json.load(open('/media/ubuntu/disk1/hyj/denC/DenseCaption/densecap_splits.json', 'r'))['train']
vgcoco_split = json.load(open('/media/ubuntu/disk1/hyj/denC/DenseCaption/vg_coco_splits.json', 'r'))['vg_coco_train']
sg_box = h5py.File(os.path.join('/media/ubuntu/disk1/hyj/denC/DenseCaption/densecap-sg100/sg100-train-all-box.h5'), 'r')

VG_split = json.load(open('/media/ubuntu/disk1/hyj/denC/DenseCaption/Untitled_Folder/image_data_v1.2.json', 'r'))

pkl = pickle.load(open('/media/ubuntu/disk1/hyj/denC/data/feat_data/detr_train_imgid2idx.pkl', 'rb'))


def cut_text(my_string, start_truncate):
    # 找到指定字符在字符串中的位置
    start_index = my_string.find(start_truncate)

    # 如果找到了指定字符，执行截断操作
    if start_index != -1:
        truncated_string = my_string[start_index:]
        return truncated_string
    return("有问题")

print('done')