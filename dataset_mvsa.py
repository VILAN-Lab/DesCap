import os
import pickle

import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import boxes as box_ops
import torchvision.transforms as transforms
from PIL import Image
from prefetch_generator import BackgroundGenerator
from utils.box_utils import generate_sg_matrix, generate_sg_matrix_box

class DataLoaderPFG(DataLoader):
    """
    Prefetch version of DataLoader: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DenseCapDataset(Dataset):

    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        return tuple(zip(*batch)) # as tuples instead of stacked tensors

    @staticmethod
    def get_transform():
        """More complicated transform utils in torchvison/references/detection/transforms.py
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def __init__(self, img_dir_root, vg_data_path, look_up_tables_path, dataset_type=None, transform=None):

        assert dataset_type in {'train', 'test', 'val'}

        super(DenseCapDataset, self).__init__()

        self.img_dir_root = img_dir_root
        self.vg_data_path = vg_data_path
        self.look_up_tables_path = look_up_tables_path
        self.dataset_type = dataset_type  # if dataset_type is None, all data will be use
        self.transform = transform

        # === load data here ====
        if self.dataset_type == 'train':
            self.feat_data = h5py.File('/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/feat_data/train.hdf5', 'r')
            self.imgid2idx = pickle.load(open('/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/feat_data/train_imgid2idx.pkl', 'rb'))
            self.densecap_split = json.load(open('./info/densecap_splits.json', 'r'))['train']
            self.vgcoco_split = json.load(open('/home/ubuntu/D/hyj/data/visual-genome-v1.2/densecap-split/vg_coco_splits.json', 'r'))['vg_coco_train']
            self.sg_box = h5py.File(os.path.join(self.img_dir_root,'densecap-sg100','sg100-train-all-box.h5'), 'r')
            self.sg_data1 = h5py.File(os.path.join(self.img_dir_root,'densecap-sg100','sg100-train-rel-0-2000.h5'), 'r')
            self.sg_data2 = h5py.File(os.path.join(self.img_dir_root, 'densecap-sg100', 'sg100-train-rel-2000-37000.h5'),'r')
            self.sg_data3 = h5py.File(os.path.join(self.img_dir_root, 'densecap-sg100', 'sg100-train-rel-37000-final.h5'),'r')
        elif self.dataset_type == 'val':
            self.feat_data = h5py.File('/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/feat_data/val.hdf5', 'r')
            self.imgid2idx = pickle.load(open('/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/feat_data/val_imgid2idx.pkl', 'rb'))
            self.sg_data = h5py.File(os.path.join(self.img_dir_root,'densecap-sg100','sg100-val.h5'), 'r')
            self.densecap_split = json.load(open('./info/densecap_splits.json', 'r'))['val']
            self.vgcoco_split = json.load(open('/home/ubuntu/D/hyj/data/visual-genome-v1.2/densecap-split/vg_coco_splits.json', 'r'))['vg_coco_val']
        elif self.dataset_type == 'test':
            self.feat_data = h5py.File('/media/ubuntu/HYJ/dataset/mvsa17024/mvsa_17024_image_visual_feat.h5', 'r')
            self.sg_data = h5py.File('/media/ubuntu/HYJ/dataset/mvsa17024/mvsa_17024_sg64.h5', 'r')
            self.densecap_split = np.load('/media/ubuntu/HYJ/dataset/mvsa17024/for_msa_2.npz')['arr_0']
            

        self.vg_data = h5py.File(vg_data_path, 'r')
        self.look_up_tables = pickle.load(open(look_up_tables_path, 'rb'))
        

    def set_dataset_type(self, dataset_type, verbose=True):

        assert dataset_type in {'train', 'test', 'val'}

        if verbose:
            print('[DenseCapDataset]: {} switch to {}'.format(self.dataset_type, dataset_type))

        self.dataset_type = dataset_type

    def __getitem__(self, idx):
        
        if self.dataset_type == 'val' or self.dataset_type == 'test':
            # get rcnn feature
            imgid = int(self.densecap_split[idx][0])
            indx = str(imgid) + '.jpg'
            img_path = os.path.join(self.img_dir_root, indx)
            rcnn_idx = int(self.feat_data['image_id'][idx])
            assert imgid == rcnn_idx

            box_num = int(self.feat_data['num_boxes'][idx])
            box = torch.from_numpy(self.feat_data['image_bb'][idx][:box_num])
            # get sg feature
            rel_num = int(self.sg_data['det_rel_num'][idx])
            rel_num = min(rel_num, 64)
            sg_obj_box = torch.from_numpy(self.sg_data['det_boxes_o_top'][idx][:rel_num]).float()
            sg_sbj_box = torch.from_numpy(self.sg_data['det_boxes_s_top'][idx][:rel_num]).float()
            iou_obj = box_ops.box_iou(sg_obj_box, box)
            iou_sbj = box_ops.box_iou(sg_sbj_box, box)
            max1, max_idx1 = torch.max(iou_obj, dim=1)
            max2, max_idx2 = torch.max(iou_sbj, dim=1)
            all_object_ind = list(set(max_idx1.tolist() + max_idx2.tolist()))
            sg_node_feat = torch.from_numpy(self.feat_data['image_features'][idx][:box_num][all_object_ind]).float()
            sg_node_box = box[all_object_ind].float()
            new_sbj_ind = torch.LongTensor([all_object_ind.index(i) for i in max_idx2])
            new_obj_ind = torch.LongTensor([all_object_ind.index(i) for i in max_idx1])
            sg_object_ind = torch.cat((new_sbj_ind.unsqueeze(0), new_obj_ind.unsqueeze(0)), dim=0)
            # get rel feature
            sg_edge_feat = torch.from_numpy(self.sg_data['rel_embeddings'][idx][:rel_num]).float()
        elif self.dataset_type == 'train':
            pass

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)


        boxes = sg_node_box
        caps = torch.zeros([3,],dtype=torch.int64)
        caps_len = torch.zeros([3,],dtype=torch.int64)

        targets = {
            'boxes': boxes,
            'caps': caps,
            'caps_len': caps_len,
            'sg_object_ind': sg_object_ind,
            'sg_node_box': sg_node_box,
            'sg_node_feat': sg_node_feat,
            'sg_edge_feat': sg_edge_feat
        }

        info = {
            'idx': idx,
            'dir': img_path,
            'file_name': indx
        }

        return img, targets, info

    def __len__(self):

        if self.dataset_type:
            return len(self.densecap_split)
        else:
            return len(self.densecap_split)
    

if __name__ == '__main__':
    from tqdm import tqdm
    IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/mvsa/MVSA_Single/data'
    VG_DATA_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-lite.h5'
    LOOK_UP_TABLES_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-dicts-lite.pkl'

    #vg_data = h5py.File(VG_DATA_PATH, 'r')

    dcd = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='test')
    l = dcd.__len__()
    i1 = dcd.__getitem__(0)

    train_loader = DataLoaderPFG(dcd, batch_size=1, shuffle=False, num_workers=4,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)
    x = 0
    y = 0
    z = 0

    for batch, (img, targets, info) in enumerate(tqdm(train_loader)):
        x += 1
    print(x,y,z)
    a = 0
        
