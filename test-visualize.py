import os
import h5py
import json
import pickle
import argparse

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from model.densecap import densecap_resnet50_fpn

from dataset import DenseCapDataset, DataLoaderPFG


def load_model(console_args):

    with open(console_args.config_json, 'r') as f:
        model_args = json.load(f)

    model = densecap_resnet50_fpn(backbone_pretrained=model_args['backbone_pretrained'],
                                  return_features=console_args.extract,
                                  feat_size=model_args['feat_size'],
                                  hidden_size=model_args['hidden_size'],
                                  max_len=model_args['max_len'],
                                  emb_size=model_args['emb_size'],
                                  rnn_num_layers=1,
                                  vocab_size=model_args['vocab_size'],
                                  fusion_type=model_args['fusion_type'],
                                  box_detections_per_img=console_args.box_per_img)

    checkpoint = torch.load(console_args.model_checkpoint)
    model.load_state_dict(checkpoint['model'])

    if console_args.verbose and 'results_on_val' in checkpoint.keys():
        print('[INFO]: checkpoint {} loaded'.format(console_args.model_checkpoint))
        print('[INFO]: correspond performance on val set:')
        for k, v in checkpoint['results_on_val'].items():
            if not isinstance(v, dict):
                print('        {}: {:.3f}'.format(k, v))

    return model


def get_image_path(console_args):

    test_split = json.load(open('./info/densecap_splits.json', 'r'))['test']

    img_list = []
    for i in test_split:
        #img_path = os.path.join(img_dir_root,look_up_tables['idx_to_directory'][i],look_up_tables['idx_to_filename'][i])
        img_list.append(console_args.img_path)

    return img_list


def img_to_tensor(img_list):

    assert isinstance(img_list, list) and len(img_list) > 0

    img_tensors = []

    for img_path in img_list:

        img = Image.open(img_path).convert("RGB")

        img_tensors.append(transforms.ToTensor()(img))

    return img_tensors


def describe_images(model, loader, device, console_args):

    assert isinstance(console_args.batch_size, int) and console_args.batch_size > 0

    all_results = []
    img_list = []

    with torch.no_grad():

        model.to(device)
        model.eval()

        for batch, (img, targets, info) in enumerate(tqdm(loader)):  # item 1 image, batch_size = 1

            img_path = os.path.join(console_args.img_path, info[0]['dir'], info[0]['file_name'])
            img_list.append(img_path)
            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k:v.to(device) for k, v in target.items()} for target in targets]
            results = model(img, targets)  # [{}, {}, ...]

            all_results.extend([{k:v.cpu() for k,v in r.items()} for r in results])

    return all_results, img_list


def save_results_to_file(img_list, all_results, console_args):

    with open(os.path.join(console_args.lut_path), 'rb') as f:
        look_up_tables = pickle.load(f)

    idx_to_token = look_up_tables['idx_to_token']

    results_dict = {}
    if console_args.extract:
        total_box = sum(len(r['boxes']) for r in all_results)
        start_idx = 0
        img_idx = 0
        h = h5py.File(os.path.join(console_args.result_dir, 'box_feats.h5'), 'w')
        h.create_dataset('feats', (total_box, all_results[0]['feats'].shape[1]), dtype=np.float32)
        h.create_dataset('boxes', (total_box, 4), dtype=np.float32)
        h.create_dataset('start_idx', (len(img_list),), dtype=np.long)
        h.create_dataset('end_idx', (len(img_list),), dtype=np.long)

    for img_path, results in zip(img_list, all_results):

        if console_args.verbose:
            print('[Result] ==== {} ====='.format(img_path))

        results_dict[img_path] = []
        for box, cap, score in zip(results['boxes'], results['caps'], results['scores']):

            r = {
                'box': [round(c, 2) for c in box.tolist()],
                'score': round(score.item(), 2),
                'cap': ' '.join(idx_to_token[idx] for idx in cap.tolist()
                                if idx_to_token[idx] not in ['<pad>', '<bos>', '<eos>'])
            }

            if console_args.verbose and r['score'] > 0.9:
                print('        SCORE {}  BOX {}'.format(r['score'], r['box']))
                print('        CAP {}\n'.format(r['cap']))

            results_dict[img_path].append(r)

        if console_args.extract:
            box_num = len(results['boxes'])
            h['feats'][start_idx: start_idx+box_num] = results['feats'].cpu().numpy()
            h['boxes'][start_idx: start_idx+box_num] = results['boxes'].cpu().numpy()
            h['start_idx'][img_idx] = start_idx
            h['end_idx'][img_idx] = start_idx + box_num - 1
            start_idx += box_num
            img_idx += 1

    if console_args.extract:
        h.close()
        # save order of img to a txt
        if len(img_list) > 1:
            with open(os.path.join(console_args.result_dir, 'feat_img_mappings.txt'), 'w') as f:
                for img_path in img_list:
                    f.writelines(os.path.split(img_path)[1] + '\n')

    if not os.path.exists(console_args.result_dir):
        os.mkdir(console_args.result_dir)
    with open(os.path.join(console_args.result_dir, 'rtrc-result-vgcoco.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)

    if console_args.verbose:
        print('[INFO] result save to {}'.format(os.path.join(console_args.result_dir, 'rtrc-wo-ir-result-v1.2.json')))
        if console_args.extract:
            print('[INFO] feats save to {}'.format(os.path.join(console_args.result_dir, 'box_feats.h5')))
            print('[INFO] order save to {}'.format(os.path.join(console_args.result_dir, 'feat_img_mappings.txt')))


def validate_box_feat(model, all_results, device, console_args):

    with torch.no_grad():

        box_describer = model.roi_heads.box_describer
        box_describer.to(device)
        box_describer.eval()

        if console_args.verbose:
            print('[INFO] start validating box features...')
        for results in tqdm(all_results, disable=not console_args.verbose):

            captions = box_describer(results['feats'].to(device))

            assert (captions.cpu() == results['caps']).all().item(), 'caption mismatch'

    if console_args.verbose:
        print('[INFO] validate box feat done, no problem')


def main(console_args):

    IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
    VG_DATA_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VGCOCO-regions-lite.h5'
    LOOK_UP_TABLES_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VGCOCO-regions-dicts-lite.pkl'

    # # 'v1.0'
    # IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
    # VG_DATA_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-lite-v1.0-new.h5'
    # LOOK_UP_TABLES_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-dicts-lite-v1.0-new.pkl'

    dcd = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='test')

    test_loader = DataLoaderPFG(dcd, batch_size=1, shuffle=False, num_workers=4,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # === prepare images ====
    #img_list = get_image_path(console_args)

    # === prepare model ====
    model = load_model(console_args)

    # === inference ====
    all_results, img_list = describe_images(model, test_loader, device, console_args)

    # === save results ====
    save_results_to_file(img_list, all_results, console_args)

    if console_args.extract and console_args.check:
        validate_box_feat(model, all_results, device, console_args)

def show_img_path():
    RESULT_JSON_PATH = './result.json'
    with open(RESULT_JSON_PATH, 'r') as f:
        results = json.load(f)

    for file_path in results.keys():
        print(file_path)

def visualize_result(image_file_path, result):

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    assert isinstance(result, list)

    img = Image.open(image_file_path)
    plt.imshow(img)
    ax = plt.gca()
    #print(result)
    for r in result:
        ax.add_patch(Rectangle((r['box'][0], r['box'][1]),
                               r['box'][2]-r['box'][0],
                               r['box'][3]-r['box'][1],
                               fill=False,
                               edgecolor='red',
                               linewidth=3))
        ax.text(r['box'][0], r['box'][1], r['cap'], style='italic', bbox={'facecolor':'white', 'alpha':0.7, 'pad':10})
    fig = plt.gcf()
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.show()

if __name__ == '__main__':
    # 'v1.0'
    # IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
    # VG_DATA_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-lite-v1.0-new.h5'
    # LOOK_UP_TABLES_PATH = '/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VG-regions-dicts-lite-v1.0-new.pkl'
    parser = argparse.ArgumentParser(description='Do dense captioning')
    parser.add_argument('--config_json', type=str,default='/home/ubuntu/D/hyj/project/densecap2/model_params/vgcoco_gcn_trans_cat_region_weight055/config.json', help="path of the json file which stored model configuration")
    parser.add_argument('--lut_path', type=str, default='/home/ubuntu/D/hyj/project/densecap-pytorch-main/data/data_new/VGCOCO-regions-dicts-lite.pkl', help='look up table path')
    parser.add_argument('--model_checkpoint', type=str, default='/home/ubuntu/D/hyj/project/densecap2/model_params/vgcoco_gcn_trans_cat_region_weight055/vgcoco_gcn_trans_cat_region_weight055_iter:380000_best_map_10.72602.pth.tar', help="path of the trained model checkpoint")
    parser.add_argument('--img_path', type=str, default='/home/ubuntu/D/hyj/data/visual-genome-v1.2',help="path of images, should be a file or a directory with only images")
    parser.add_argument('--result_dir', type=str, default='./test-result',
                        help="path of the directory to save the output file")
    parser.add_argument('--box_per_img', type=int, default=100, help='max boxes to describe per image')
    parser.add_argument('--batch_size', type=int, default=1, help="useful when img_path is a directory")
    parser.add_argument('--extract', action='store_true', help='whether to extract features')
    parser.add_argument('--cpu', action='store_true', help='whether use cpu to compute')
    parser.add_argument('--verbose', action='store_true', help='whether output info')
    parser.add_argument('--check', action='store_true', help='whether to validate box feat by regenerate sentences')
    args = parser.parse_args()

    #main(args)

    RESULT_JSON_PATH = './test-result/rtrc-result-v1.2.json'
    with open(RESULT_JSON_PATH, 'r') as f:
        results = json.load(f)

    #IMG_FILE_PATH = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K/2335283.jpg'
    #reg_data = results[IMG_FILE_PATH]
    all_images = []
    for file_path in results.keys():
        #print(file_path)
        all_images.append(file_path)
    print('total_test_images:{}'.format(len(all_images)))
    for i in range(80,81):
        
        #IMG_FILE_PATH = all_images[i]

        IMG_FILE_PATH = '/home/ubuntu/D/hyj/data/visual-genome-v1.2/VG_100K/2322710.jpg'

        print(IMG_FILE_PATH)
        start, TO_K = 0,10

        visualize_result(IMG_FILE_PATH, results[IMG_FILE_PATH][start:TO_K])
