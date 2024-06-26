import os
import json

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Subset
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch.utils.tensorboard import SummaryWriter

#from utils.data_loader import DenseCapDataset, DataLoaderPFG
from dataset import DenseCapDataset, DataLoaderPFG
from model.densecap import densecap_resnet50_fpn

from apex import amp
import math

from evaluate import quality_check, quantity_check
from utils.box_utils import get_std_opt, set_lr, get_lr

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

np.random.seed(555)  # 42
torch.manual_seed(555)
torch.cuda.manual_seed(555)

MAX_EPOCHS = 15
USE_TB = True
USE_TB2 = False
Load_Model = True
CONFIG_PATH = './model_params'

### v1.2 ###
MODEL_NAME = 'v1.2_gcn_trans_cat_region_weight055'
IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
VG_DATA_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-dicts-lite.pkl'

# ### vg-coco ###
# MODEL_NAME = 'vgcoco_gcn_trans_cat_region_weight055'
# IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
# VG_DATA_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VGCOCO-regions-lite.h5'
# LOOK_UP_TABLES_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VGCOCO-regions-dicts-lite.pkl'

### v1.0 ###
# MODEL_NAME = 'rrtransformer-v1.0-2'
# IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
# VG_DATA_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-lite-v1.0-new.h5'
# LOOK_UP_TABLES_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-dicts-lite-v1.0-new.pkl'

MAX_TRAIN_IMAGE = -1  # if -1, use all images in train set
MAX_VAL_IMAGE = -1


def set_args():

    args = dict()

    args['backbone_pretrained'] = True
    args['return_features'] = False

    # Caption parameters
    args['feat_size'] = 2048
    args['hidden_size'] = 512
    args['max_len'] = 16
    args['emb_size'] = 512
    args['vocab_size'] = 10629  # V1.2
    #args['vocab_size'] = 7869  # coco
    #args['vocab_size'] = 10509  # v1.0
    args['fusion_type'] = 'init_inject'

    # Training Settings
    args['detect_loss_weight'] = 0.5  # 1.
    args['caption_loss_weight'] = 1.
    args['lr'] = 2e-5
    args['caption_lr'] = 2e-5
    args['weight_decay'] = 0.
    args['batch_size'] = 1  # 4
    args['use_pretrain_fasterrcnn'] = False
    args['box_detections_per_img'] = 64   # use in val/test

    # learning rate
    args['learning_rate'] = 2e-5
    args['learning_rate_decay_start'] = 0
    args['learning_rate_decay_every'] = 1
    args['learning_rate_decay_rate'] = 0.8

    # optim
    args['optim'] = 'AdamW'

    if not os.path.exists(os.path.join(CONFIG_PATH, MODEL_NAME)):
        os.mkdir(os.path.join(CONFIG_PATH, MODEL_NAME))
    with open(os.path.join(CONFIG_PATH, MODEL_NAME, 'config.json'), 'w') as f:
        json.dump(args, f, indent=2)

    return args


def save_model(model, optimizer, results_on_val, iter_counter, flag=None):

    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),     # optimizer.state_dict()
             # 'amp': amp_.state_dict(),
             'results_on_val':results_on_val,
             'iterations': iter_counter}
    if isinstance(flag, str):
        filename = os.path.join('model_params', MODEL_NAME, '{}_{}.pth.tar'.format(MODEL_NAME, flag))
    else:
        filename = os.path.join('model_params', MODEL_NAME, '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

def save_model2(model, optimizer, results_on_test, iter_counter, flag=None):

    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),     # optimizer.state_dict()
             # 'amp': amp_.state_dict(),
             'results_on_test':results_on_test,
             'iterations': iter_counter}
    if isinstance(flag, str):
        filename = os.path.join('model_params', MODEL_NAME, '{}_{}.pth.tar'.format(MODEL_NAME, flag))
    else:
        filename = os.path.join('model_params', MODEL_NAME, '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)

def load_model(p1, p2, model_name, devic):

    with open(p1, 'r') as f:
        model_args = json.load(f)

    model = densecap_resnet50_fpn(backbone_pretrained=model_args['backbone_pretrained'],
                                  feat_size=model_args['feat_size'],
                                  hidden_size=model_args['hidden_size'],
                                  max_len=model_args['max_len'],
                                  emb_size=model_args['emb_size'],
                                  vocab_size=model_args['vocab_size'],
                                  fusion_type=model_args['fusion_type'],
                                  box_detections_per_img=model_args['box_detections_per_img'])
    
    optimizer = torch.optim.AdamW([{'params': (para for name, para in model.named_parameters()
                                    if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in model.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr': model_args['caption_lr']}],
                                  lr=model_args['learning_rate'], weight_decay=model_args['weight_decay'])

    checkpoint = torch.load(p2, map_location=devic)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(devic)
    iters = checkpoint['iterations']
    bestmap = checkpoint['results_on_val']['map']

    if 'results_on_val' in checkpoint.keys():
        print('[INFO]: checkpoint {} loaded'.format(model_name))
        print('[INFO]: correspond performance on val set:')
        for k, v in checkpoint['results_on_val'].items():
            if not isinstance(v, dict):
                print('        {}: {:.7f}'.format(k, v))
    else:
        print('[INFO]: checkpoint {} loaded'.format(model_name))
        print('[INFO]: correspond performance on test set:')
        for k, v in checkpoint['results_on_test'].items():
            if not isinstance(v, dict):
                print('        {}: {:.7f}'.format(k, v))

    return model, optimizer, iters, bestmap

def train(args):

    iter_counter = 0
    best_map = 0

    if Load_Model:
        checkpoint = 'v1.2_gcn_trans_cat_region_weight055_iter:550001_best_map_11.81112.pth.tar'
        print('Model start training from {}'.format(checkpoint))
        p1 = './model_params/v1.2_gcn_trans_cat_region_weight055/config.json'
        p2 = './model_params/v1.2_gcn_trans_cat_region_weight055/v1.2_gcn_trans_cat_region_weight055_iter:550001_best_map_11.81112.pth.tar'
        model, optimizer, iter_counter, best_map = load_model(p1, p2, checkpoint, device)
        iter_counter += 1
        
    else:
        print('Model {} start training...'.format(MODEL_NAME))

        model = densecap_resnet50_fpn(backbone_pretrained=args['backbone_pretrained'],
                                  feat_size=args['feat_size'],
                                  hidden_size=args['hidden_size'],
                                  max_len=args['max_len'],
                                  emb_size=args['emb_size'],
                                  vocab_size=args['vocab_size'],
                                  fusion_type=args['fusion_type'],
                                  box_detections_per_img=args['box_detections_per_img'])

        optimizer = torch.optim.AdamW([{'params': (para for name, para in model.named_parameters()
                                        if para.requires_grad and 'box_describer' not in name)},
                                    {'params': (para for para in model.roi_heads.box_describer.parameters()
                                                if para.requires_grad), 'lr': args['caption_lr']}],
                                    lr=args['learning_rate'], weight_decay=args['weight_decay'])

    
    if args['use_pretrain_fasterrcnn']:
        model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).backbone.state_dict(), strict=False)
        model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).rpn.state_dict(), strict=False)

    #model = nn.DataParallel(model, device_ids=device_id)
    model.to(device)
    
    # optimizer = torch.optim.SGD([{'params': (para for name, para in model.named_parameters()
    #                                 if para.requires_grad and 'box_describer' not in name)},
    #                               {'params': (para for para in model.roi_heads.box_describer.parameters()
    #                                           if para.requires_grad), 'lr': args['caption_lr']}],
    #                               lr=args['learning_rate'], weight_decay=args['weight_decay'],momentum=0.9)
    
    #optimizer = get_std_opt(optimizer, factor=args['noamopt_factor'], warmup=args['noamopt_warmup'])
    #optimizer._step = iter_counter

    #if multi_gpu:
        #optimizer = nn.DataParallel(optimizer, device_ids=device_id)
    
    # apex initialization
    opt_level = 'O0'
    #model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    # ref: https://github.com/NVIDIA/apex/issues/441
    #model.roi_heads.box_roi_pool.forward = amp.half_function(model.roi_heads.box_roi_pool.forward)

    
    train_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='train')
    val_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='val')
    test_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='test')

    idx_to_token = train_set.look_up_tables['idx_to_token']

    if MAX_TRAIN_IMAGE > 0:
        train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))
    if MAX_VAL_IMAGE > 0:
        val_set = Subset(val_set, range(MAX_VAL_IMAGE))

    train_loader = DataLoaderPFG(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)


    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    for epoch in range(MAX_EPOCHS):
        if Load_Model:
            epo = math.floor(iter_counter / 77398)
            epoch += epo

        for batch, (img, targets, info) in enumerate(train_loader):  # item 1 image, batch_size = 1

            # Assign the learning rate
            if epoch > args['learning_rate_decay_start'] and args['learning_rate_decay_start'] >= 0:
                frac = (epoch - args['learning_rate_decay_start']) // args['learning_rate_decay_every']
                decay_factor = args['learning_rate_decay_rate']  ** frac
                current_lr = args['learning_rate'] * decay_factor
            else:
                current_lr = args['learning_rate']
            set_lr(optimizer, current_lr) # set the decayed rate

            img = [img_tensor.to(device) for img_tensor in img]  # [tensor0, tensor1, tensor2, tensor3], tensor0:[3,375,500]...
            targets = [{k:v.to(device) for k, v in target.items()} for target in targets]
            # [{'boxes':tensor[48,4], 'caps':tensor[48,17]}, 'caps_len':tensor[48], {}, {}, {}]

            results = quantity_check(model, test_set, idx_to_token, device, max_iter=-1, verbose=True)
            model.train()
            #model.eval()
            losses = model(img, targets)
            #print(losses)

            rpn_loss =  losses['loss_objectness'] + losses['loss_rpn_box_reg']  # rpn loss
            roi_loss = losses['loss_classifier'] + losses['loss_box_reg']
            caption_loss = losses['loss_caption']
            iou_loss = losses['iou_loss']


            total_loss = args['detect_loss_weight'] * (rpn_loss+roi_loss) + args['caption_loss_weight'] * caption_loss
            total_loss2 = args['detect_loss_weight'] * (rpn_loss+roi_loss) + args['caption_loss_weight'] * caption_loss * iou_loss
            # record loss
            # if USE_TB:
            #     writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
            #     writer.add_scalar('details/caption_loss', caption_loss.item(), iter_counter)
            #     writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
            #     writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
            #     writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
            #     writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)

            #     optimizer_lr = get_lr(optimizer)
            #     writer.add_scalar('learning_rate', optimizer_lr, iter_counter)

            if iter_counter % 10 == 0:
                if USE_TB2:
                    writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
                    writer.add_scalar('details/caption_loss', caption_loss.item(), iter_counter)
                    writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
                    writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
                    writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
                    writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)
                    optimizer_lr = get_lr(optimizer)
                    writer.add_scalar('learning_rate', optimizer_lr, iter_counter)

                print("[epoch:{}][batch:{}]\ntotal_loss {:.5f}".format(epoch, batch, total_loss.item()))
                for k, v in losses.items():
                    print(" <{}> {:.5f}".format(k, v))

            optimizer.zero_grad()
            total_loss.backward()
            # apex backward
            #with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                #scaled_loss.backward()
            optimizer.step()

            if iter_counter >= 500000 and iter_counter % 1 == 0:
                try:
                    results = quantity_check(model, val_set, idx_to_token, device, max_iter=-1, verbose=True)
                    
                    if results['map'] > best_map:
                        best_map = results['map']
                        best_map_flag = str(best_map * 100)[:8]
                        info_flag = 'iter:' + str(iter_counter)
                        save_flag = info_flag + '_best_map_' + best_map_flag
                        save_model(model, optimizer, results, iter_counter, flag=save_flag)

                    if USE_TB:
                        writer.add_scalar('metric/map', results['map'], iter_counter)
                        writer.add_scalar('metric/det_map', results['detmap'], iter_counter)

                except AssertionError as e:
                    print('[INFO]: evaluation failed at epoch {}'.format(epoch))
                    print(e)

            iter_counter += 1

    save_model(model, optimizer, results, iter_counter, flag='end')

    if USE_TB:
        writer.close()

def eval_on_test(args):

    iter_counter = 0
    best_map = 0

    if Load_Model:
        checkpoint = 'v1.2_gcn_trans_cat_region_weight055_iter:520000_best_map_11.80182.pth.tar'
        print('Model start training from {}'.format(checkpoint))
        p1 = './model_params/v1.2_gcn_trans_cat_region_weight055/config.json'
        p2 = './model_params/v1.2_gcn_trans_cat_region_weight055/v1.2_gcn_trans_cat_region_weight055_iter:520000_best_map_11.80182.pth.tar'
        model, optimizer, iter_counter, best_map = load_model(p1, p2, checkpoint, device)
        iter_counter += 1
        
    else:
        print('Model {} start training...'.format(MODEL_NAME))

        model = densecap_resnet50_fpn(backbone_pretrained=args['backbone_pretrained'],
                                  feat_size=args['feat_size'],
                                  hidden_size=args['hidden_size'],
                                  max_len=args['max_len'],
                                  emb_size=args['emb_size'],
                                  vocab_size=args['vocab_size'],
                                  fusion_type=args['fusion_type'],
                                  box_detections_per_img=args['box_detections_per_img'])

        optimizer = torch.optim.AdamW([{'params': (para for name, para in model.named_parameters()
                                        if para.requires_grad and 'box_describer' not in name)},
                                    {'params': (para for para in model.roi_heads.box_describer.parameters()
                                                if para.requires_grad), 'lr': args['caption_lr']}],
                                    lr=args['learning_rate'], weight_decay=args['weight_decay'])

    
    if args['use_pretrain_fasterrcnn']:
        model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).backbone.state_dict(), strict=False)
        model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).rpn.state_dict(), strict=False)

    #model = nn.DataParallel(model, device_ids=device_id)
    model.to(device)

    
    train_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='train')
    test_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='test')

    idx_to_token = train_set.look_up_tables['idx_to_token']

    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    for epoch in range(MAX_EPOCHS):

        results = quantity_check(model, test_set, idx_to_token, device, max_iter=-1, verbose=True)
                    
        if results['map'] > best_map:
            best_map = results['map']
            best_map_flag = str(best_map * 100)[:8]
            info_flag = 'iter:' + str(iter_counter)
            save_flag = info_flag + '_best_map_test_' + best_map_flag
            save_model2(model, optimizer, results, iter_counter, flag=save_flag)

        if USE_TB:
            writer.add_scalar('metric/map', results['map'], iter_counter)
            writer.add_scalar('metric/det_map', results['detmap'], iter_counter)

    if USE_TB:
        writer.close()


if __name__ == '__main__':

    args = set_args()
    #train(args)
    eval_on_test(args)
