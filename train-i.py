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

from evaluate import quality_check, quantity_check
from utils.box_utils import get_std_opt, set_lr, get_lr

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

np.random.seed(55)  # 42
torch.manual_seed(55)
torch.cuda.manual_seed(55)

MAX_EPOCHS = 8
USE_TB = True
CONFIG_PATH = './model_params'
MODEL_NAME = 'dr_transformer_bz_1_epoch_8-p-i'
IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
VG_DATA_PATH = './data/data_new/VG-regions-lite.h5'
LOOK_UP_TABLES_PATH = './data/data_new/VG-regions-dicts-lite.pkl'
MAX_TRAIN_IMAGE = -1  # if -1, use all images in train set
MAX_VAL_IMAGE = -1


def set_args():

    args = dict()

    args['backbone_pretrained'] = True
    args['return_features'] = False

    # Caption parameters
    args['feat_size'] = 4096
    args['hidden_size'] = 512
    args['max_len'] = 16
    args['emb_size'] = 512
    args['rnn_num_layers'] = 1
    args['vocab_size'] = 10629
    args['fusion_type'] = 'init_inject'

    # Training Settings
    args['detect_loss_weight'] = 1.
    args['caption_loss_weight'] = 1.
    args['lr'] = 2e-5
    args['caption_lr'] = 2e-5
    args['weight_decay'] = 0.
    args['batch_size'] = 1  # 4
    args['use_pretrain_fasterrcnn'] = True
    args['box_detections_per_img'] = 50   # use in val/test

    # learning rate
    args['learning_rate'] = 2e-5
    args['learning_rate_decay_start'] = 0
    args['learning_rate_decay_every'] = 1
    args['learning_rate_decay_rate'] = 0.8

    # optim
    args['noamopt_factor'] = 1
    args['noamopt_warmup'] = 10000

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
        filename = os.path.join('model_params', '{}_{}.pth.tar'.format(MODEL_NAME, flag))
    else:
        filename = os.path.join('model_params', '{}.pth.tar'.format(MODEL_NAME))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)


def train(args):

    print('Model {} start training...'.format(MODEL_NAME))
    iter_counter = 0
    best_map = 0.

    model = densecap_resnet50_fpn(backbone_pretrained=args['backbone_pretrained'],
                                  feat_size=args['feat_size'],
                                  hidden_size=args['hidden_size'],
                                  max_len=args['max_len'],
                                  emb_size=args['emb_size'],
                                  rnn_num_layers=args['rnn_num_layers'],
                                  vocab_size=args['vocab_size'],
                                  fusion_type=args['fusion_type'],
                                  box_detections_per_img=args['box_detections_per_img'])
    if args['use_pretrain_fasterrcnn']:
        model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).backbone.state_dict(), strict=False)
        model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).rpn.state_dict(), strict=False)

    #model = nn.DataParallel(model, device_ids=device_id)
    model.to(device)

    optimizer = torch.optim.Adam([{'params': (para for name, para in model.named_parameters()
                                    if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in model.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr': args['caption_lr']}],
                                  lr=args['learning_rate'], weight_decay=args['weight_decay'])
    
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
    idx_to_token = train_set.look_up_tables['idx_to_token']

    if MAX_TRAIN_IMAGE > 0:
        train_set = Subset(train_set, range(MAX_TRAIN_IMAGE))
    if MAX_VAL_IMAGE > 0:
        val_set = Subset(val_set, range(MAX_VAL_IMAGE))

    train_loader = DataLoaderPFG(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)


    # use tensorboard to track the loss
    if USE_TB:
        writer = SummaryWriter()

    for epoch in range(MAX_EPOCHS):

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

            model.train()
            #model.eval()
            losses = model(img, targets)
            #print(losses)

            detect_loss =  losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                           losses['loss_classifier'] + losses['loss_box_reg']
            caption_loss = losses['loss_caption']


            total_loss = args['detect_loss_weight'] * detect_loss + args['caption_loss_weight'] * caption_loss
            
            # record loss
            if USE_TB:
                writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/detect_loss', detect_loss.item(), iter_counter)
                writer.add_scalar('batch_loss/caption_loss', caption_loss.item(), iter_counter)

                writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
                writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
                writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
                writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)

                optimizer_lr = get_lr(optimizer)
                writer.add_scalar('learning_rate', optimizer_lr, iter_counter)

            if iter_counter % 5 == 0:
                print("[epoch:{}][batch:{}]\ntotal_loss {:.3f}".format(epoch, batch, total_loss.item()))
                for k, v in losses.items():
                    print(" <{}> {:.3f}".format(k, v))

            optimizer.zero_grad()
            total_loss.backward()
            # apex backward
            #with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                #scaled_loss.backward()
            optimizer.step()

            if iter_counter > 0 and iter_counter % 20000 == 0:
                try:
                    results = quantity_check(model, val_set, idx_to_token, device, max_iter=-1, verbose=True)
                    if results['map'] > best_map:
                        best_map = results['map']
                        best_map_flag = str(best_map * 100)[:8]
                        info_flag = str(epoch) + str(batch)
                        save_flag = 'best_map_' + best_map_flag
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


if __name__ == '__main__':

    args = set_args()
    train(args)
