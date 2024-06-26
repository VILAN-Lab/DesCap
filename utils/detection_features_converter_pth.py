"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import json
import pickle
import numpy as np
from tqdm import tqdm
import torch


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
# infile = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/genome_resnet101_faster_rcnn_genome.tsv'
# train_data_file = '../data/feat_data/train.hdf5'
# val_data_file = '../data/feat_data/val.hdf5'
# test_data_file = '../data/feat_data/test.hdf5'
# train_indices_file = '../data/feat_data/train_imgid2idx.pkl'
# val_indices_file = '../data/feat_data/val_imgid2idx.pkl'
# test_indices_file = '../data/feat_data/test_imgid2idx.pkl'

infile = '/media/ubuntu/disk1/hyj/denC/DenseCaption/detr.tsv'
train_data_file = '/media/ubuntu/disk1/hyj/denC/data/feat_data/detr_train.hdf5'
val_data_file = '/media/ubuntu/disk1/hyj/denC/data/feat_data/detr_val.hdf5'
test_data_file = '/media/ubuntu/disk1/hyj/denC/data/feat_data/detr_test.hdf5'
train_indices_file = '/media/ubuntu/disk1/hyj/denC/data/feat_data/detr_train_imgid2idx.pkl'
val_indices_file = '/media/ubuntu/disk1/hyj/denC/data/feat_data/detr_val_imgid2idx.pkl'
test_indices_file = '/media/ubuntu/disk1/hyj/denC/data/feat_data/detr_test_imgid2idx.pkl'

feature_length = 256 # 2048
num_fixed_boxes = 100


if __name__ == '__main__':

    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")
    h_test = h5py.File(test_data_file, "w")

    splits = json.load(open('/media/ubuntu/disk1/hyj/denC/DenseCaption/densecap_splits.json', 'r'))
    train_imgids = splits['train']
    val_imgids = splits['val']
    test_imgids = splits['test']

    train_indices = {}
    val_indices = {}
    test_indices = {}

    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    train_num_boxes = h_train.create_dataset('num_boxes', (len(train_imgids),), 'f')
    #train_spatial_img_features = h_train.create_dataset('spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    val_num_boxes = h_val.create_dataset('num_boxes', (len(val_imgids),), 'f')
    #val_spatial_img_features = h_val.create_dataset('spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')

    test_img_bb = h_test.create_dataset(
        'image_bb', (len(test_imgids), num_fixed_boxes, 4), 'f')
    test_img_features = h_test.create_dataset(
        'image_features', (len(test_imgids), num_fixed_boxes, feature_length), 'f')
    test_num_boxes = h_test.create_dataset('num_boxes', (len(test_imgids),), 'f')
    #test_spatial_img_features = h_test.create_dataset('spatial_features', (len(test_imgids), num_fixed_boxes, 6), 'f')

    train_counter = 0
    val_counter = 0
    test_counter = 0
    print('reading pth...')
    pth = torch.load('/media/ubuntu/disk1/hyj/denC/DenseCaption/detr2.pth')
    print('reading pth done')
    no = 0
    for item in tqdm(pth):
        # print(item)
        item['num_boxes'] = int(item['num_boxes'])
        box_num = item['num_boxes']
        image_id = int(item['image_id'])
        image_w = float(item['image_w'])
        image_h = float(item['image_h'])
        bboxes = item['boxes'].reshape((item['num_boxes'], -1))

        box_width = bboxes[:, 2] - bboxes[:, 0]
        box_height = bboxes[:, 3] - bboxes[:, 1]
        scaled_width = box_width / image_w
        scaled_height = box_height / image_h
        scaled_x = bboxes[:, 0] / image_w
        scaled_y = bboxes[:, 1] / image_h

        box_width = box_width[..., np.newaxis]
        box_height = box_height[..., np.newaxis]
        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]

        spatial_features = np.concatenate(
            (scaled_x,
                scaled_y,
                scaled_x + scaled_width,
                scaled_y + scaled_height,
                scaled_width,
                scaled_height),
            axis=1)


        if image_id in train_imgids:
            train_imgids.remove(image_id)
            train_indices[image_id] = train_counter
            train_num_boxes[train_counter] = box_num
            train_img_bb[train_counter, :box_num, :] = bboxes
            train_img_features[train_counter, :box_num, :] = item['features'].reshape((item['num_boxes'], -1))
            #train_spatial_img_features[train_counter, :box_num, :] = spatial_features
            train_counter += 1
        elif image_id in val_imgids:
            val_imgids.remove(image_id)
            val_indices[image_id] = val_counter
            val_num_boxes[val_counter] = box_num
            val_img_bb[val_counter, :box_num, :] = bboxes
            val_img_features[val_counter, :box_num, :] = item['features'].reshape((item['num_boxes'], -1))
            #val_spatial_img_features[val_counter, :box_num, :] = spatial_features
            val_counter += 1
        elif image_id in test_imgids:
            test_imgids.remove(image_id)
            test_indices[image_id] = test_counter
            test_num_boxes[test_counter] = box_num
            test_img_bb[test_counter, :box_num, :] = bboxes
            test_img_features[test_counter, :box_num, :] = item['features'].reshape((item['num_boxes'], -1))
            #test_spatial_img_features[test_counter, :box_num, :] = spatial_features
            test_counter += 1
        else:
            #assert False, 'Unknown image id: %d' % image_id
            no += 1

    '''
    print("reading tsv...")
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        no = 0
        for item in tqdm(reader):
            item['num_boxes'] = int(item['num_boxes'])
            box_num = item['num_boxes']
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(item['boxes'].decode('utf-8')),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_num_boxes[train_counter] = box_num
                train_img_bb[train_counter, :box_num, :] = bboxes
                train_img_features[train_counter, :box_num, :] = np.frombuffer(
                    base64.decodestring(item['features'].encode('utf-8')),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                #train_spatial_img_features[train_counter, :box_num, :] = spatial_features
                train_counter += 1
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_num_boxes[val_counter] = box_num
                val_img_bb[val_counter, :box_num, :] = bboxes
                val_img_features[val_counter, :box_num, :] = np.frombuffer(
                    base64.decodestring(item['features'].encode('utf-8')),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                #val_spatial_img_features[val_counter, :box_num, :] = spatial_features
                val_counter += 1
            elif image_id in test_imgids:
                test_imgids.remove(image_id)
                test_indices[image_id] = test_counter
                test_num_boxes[test_counter] = box_num
                test_img_bb[test_counter, :box_num, :] = bboxes
                test_img_features[test_counter, :box_num, :] = np.frombuffer(
                    base64.decodestring(item['features'].encode('utf-8')),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                #test_spatial_img_features[test_counter, :box_num, :] = spatial_features
                test_counter += 1
            else:
                #assert False, 'Unknown image id: %d' % image_id
                no += 1
    '''
    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')
    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')
    if len(test_imgids) != 0:
        print('Warning: test_image_ids is not empty')

    pickle.dump(train_indices, open(train_indices_file, 'wb'))
    pickle.dump(val_indices, open(val_indices_file, 'wb'))
    pickle.dump(test_indices, open(test_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    h_test.close()
    print('rest '+str(no)+' images')
    print("done!")
