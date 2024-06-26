import torch
from tqdm import tqdm

#from utils.data_loader import DenseCapDataset, DataLoaderPFG
from dataset import DenseCapDataset, DataLoaderPFG
from model.evaluator import DenseCapEvaluator
from model.densecap import densecap_resnet50_fpn
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn


def quality_check(model, dataset, idx_to_token, device, max_iter=-1):

    model.to(device)
    data_loader = DataLoaderPFG(dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    print('[quality check]')
    for i, (img, targets, info) in enumerate(data_loader):

        img = [img_tensor.to(device) for img_tensor in img]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        with torch.no_grad():
            model.eval()
            model.return_features = False
            detections = model(img)

        for j in range(len(targets)):
            print('<{}>'.format(info[j]['file_name']))
            print('=== ground truth ===')
            for box, cap, cap_len in zip(targets[j]['boxes'], targets[j]['caps'], targets[j]['caps_len']):
                print('box:', box.tolist())
                print('len:', cap_len.item())
                print('cap:', ' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>'))
                print('-'*20)

            print('=== predict ===')
            for box, cap, score in zip(detections[j]['boxes'], detections[j]['caps'], detections[j]['scores']):
                print('box:', [round(c, 2) for c in box.tolist()])
                print('score:', round(score.item(), 2))
                print('cap:', ' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>'))
                print('-'*20)

        if i >= max_iter > 0:
            break


def quantity_check(model, dataset, idx_to_token, device, max_iter=-1, verbose=True):

    model.to(device)
    data_loader = DataLoaderPFG(dataset, batch_size=1, shuffle=False, num_workers=0,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)

    evaluator = DenseCapEvaluator(list(model.roi_heads.box_describer.special_idx.keys()))

    print('[quantity check]')
    for i, (img, targets, info) in tqdm(enumerate(data_loader), total=len(data_loader)):

        img = [img_tensor.to(device) for img_tensor in img]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        with torch.no_grad():
            model.eval()
            model.return_features = False
            detections = model(img,targets)

        for j in range(len(targets)):
            scores = detections[j]['scores']
            boxes = detections[j]['boxes']
            text = [' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>')
                    for cap in detections[j]['caps']]
            target_boxes = targets[j]['boxes']
            target_text = [' '.join(idx_to_token[idx] for idx in cap.tolist() if idx_to_token[idx] != '<pad>')
                    for cap in targets[j]['caps']]
            img_id = info[j]['file_name']

            evaluator.add_result(scores, boxes, text, target_boxes, target_text, img_id)


        if i >= max_iter > 0:
            break

    results = evaluator.evaluate(verbose)
    if verbose:
        print('MAP: {:.6f} DET_MAP: {:.6f}'.format(results['map'], results['detmap']))

    return results

if __name__ == '__main__':
    from torch.utils.data.dataset import Subset
    CONFIG_PATH = './model_params'
    MODEL_NAME = 'train_all_val_all_bz_2_epoch_10_inject_init'
    IMG_DIR_ROOT = '/home/ubuntu/D/hyj/data/visual-genome-v1.2'
    VG_DATA_PATH = './data/data_new/VG-regions-lite.h5'
    LOOK_UP_TABLES_PATH = './data/data_new/VG-regions-dicts-lite.pkl'
    MAX_TRAIN_IMAGE = -1  # if -1, use all images in train set
    MAX_VAL_IMAGE = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densecap_resnet50_fpn(backbone_pretrained=True,
                                  feat_size=4096,
                                  hidden_size=512,
                                  max_len=16,
                                  emb_size=512,
                                  rnn_num_layers=1,
                                  vocab_size=10629,
                                  fusion_type='init_inject',
                                  box_detections_per_img=50)
    model.backbone.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).backbone.state_dict(), strict=False)
    model.rpn.load_state_dict(fasterrcnn_resnet50_fpn(pretrained=True).rpn.state_dict(), strict=False)
    model.cuda()

    train_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='train')
    val_set = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH, dataset_type='val')
    idx_to_token = train_set.look_up_tables['idx_to_token']
    val_set = Subset(val_set, range(MAX_VAL_IMAGE))

    resual = quantity_check(model,val_set,idx_to_token,device,max_iter=-1,verbose=True)
    print(resual)