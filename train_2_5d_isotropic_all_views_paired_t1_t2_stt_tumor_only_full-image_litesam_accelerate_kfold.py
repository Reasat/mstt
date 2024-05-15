#!/usr/bin/env python
# coding: utf-8

# In[114]:
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import sys
import random
import nibabel as nib
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import time
from PIL import Image
from skimage.util import montage
import cv2 
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
# from backboned_unet import Unet
import monai
import pytorch_lightning as pl
from utils import MyxoidDataset_T1_from_T2_path, seed_everything
from sklearn.model_selection import train_test_split
from logger import Logger
import pandas as pd
# import segmentation_models_pytorch as smp

from segment_anything import sam_model_registry
from sam_surgery import MedSAM_Lite_6ch, build_medsam_lite

import torch.multiprocessing as mp
from accelerate import PartialState, prepare_pippy
from accelerate.utils import set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs

parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type = float, default = 0.25)
parser.add_argument('--epoch', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 0.0001)
parser.add_argument('--save_every', type = int, default = 1)
parser.add_argument('--print_every', type = int, default = 100)
parser.add_argument('--gpus', default = '4')
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--num_workers', type = int, default = 8)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--resume_from')#, default='model_data/2024-01-08-16-21-57/2024-01-08-16-21-57_epoch-49_loss-0.1278.ckpt')
parser.add_argument('--img_type', type = str, default='T2')
parser.add_argument('--input_size', type = int, default=256)
parser.add_argument('--thresh_tumor', type = int,  default = 100)
parser.add_argument('--train_mode')
parser.add_argument('--loss_type')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' #or CUBLAS_WORKSPACE_CONFIG=:16:8

seed_everything(args.seed)


args = parser.parse_args()

### LOGGER ###
TIME_STAMP=time.strftime('%Y-%m-%d-%H-%M-%S')
dir_output = os.path.join('/tank/data/Project-reasatt/soft_tissue_tumor/model_data', TIME_STAMP)
os.makedirs(dir_output, exist_ok = True)
sys.stdout = Logger(os.path.join(dir_output,TIME_STAMP+'.log'))

print(TIME_STAMP)
print('filename', __file__)
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))
    
    filepath_cfg = os.path.join(dir_output, TIME_STAMP+'.cfg')
    with open(filepath_cfg,'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

############ Multi-GPU #############
sys.setrecursionlimit(2500)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
        cpu=False, 
        mixed_precision='fp16', 
        project_dir = dir_output,
        #split_batches = True, 
        kwargs_handlers=[ddp_kwargs]
        )
device = accelerator.device

def format_mask_path(p):
    dir = p.split(os.sep)[-3]
   
    for view in ['axial', 'sagittal', 'coronal']:
       if view in dir:
           mask_dir = replace_map[view]

    pid = p.split(os.sep)[-1][:3]
    modality = df_consensus.groupby('PID').get_group(pid)['Modality'].item()
    is_multi_annotation= df_consensus.groupby('PID').get_group(pid)['Multiple_Annotations'].item()
#    print(p, is_multi_annotation) 
    p_mask = p.replace(dir, mask_dir).replace('T1',modality).replace('T2', modality) 

    if is_multi_annotation:
        p_mask = p_mask.replace('_'+modality, '_'+modality+'_J')
#        print(p_mask)

    assert os.path.exists(p_mask), (p, p_mask)
    assert 'mask' in p_mask, (p, p_mask)
    return p_mask


df_consensus = pd.read_csv('/scratch/reasatt/Radiomics_Research/consensus_information.csv')

###TRAIN TEST PARTITION###
paths = glob('/scratch/reasatt/Radiomics_Research/image_2_5d_isotropic_*_stride-1/*/*')
paths = [p for p in paths if args.img_type in p]

print('check patients for issues',
        set(df_consensus.PID.to_list()).difference(set(np.unique([p.split(os.sep)[-1][:3] for p in paths])))
        )
df_meta = pd.DataFrame({
    'slice_distribution': [p.split(os.sep)[-1][0] for p in paths]
    }
    )
print(df_meta.value_counts())
df_meta = pd.DataFrame(
        {
            'patient_distribution': [pid[0] for pid in np.unique([p.split(os.sep)[-1][:3] for p in paths])]

        }
    )
print(df_meta.value_counts())



# take only slices with tumor
print('filtering tumors ...')
df_tumor_views = {}
for view in ['axial', 'sagittal', 'coronal']:
    df_tumor_count = pd.read_csv('/scratch/reasatt/Radiomics_Research/mask_isotropic_{}_tumor_pixel_count.csv'.format(view))
    df_tumor_count = df_tumor_count[df_tumor_count['tumor_pixels']>args.thresh_tumor].reset_index(drop=True)
    df_tumor_count['PID'] = df_tumor_count.flnames.apply(lambda x: x.split('_')[0])
    df_tumor_count['InstanceNumber'] = df_tumor_count.flnames.apply(lambda x: x.split('_')[-1].split('.')[0])
    df_tumor_views[view] = df_tumor_count.groupby(['PID', 'InstanceNumber']).groups.keys()
  
paths_filtered = []
for p in tqdm(paths):
    for view in ['axial', 'sagittal', 'coronal']:
        if view in p:
            pid = p.split(os.sep)[-1][:3]
            instanceNumber = p.split(os.sep)[-1].split('_')[-1].split('.')[0]
            if (pid, instanceNumber) in df_tumor_views[view]:
               paths_filtered.append(p)
paths = paths_filtered
print('check patients for issues',
        set(df_consensus.PID.to_list()).difference(set(np.unique([p.split(os.sep)[-1][:3] for p in paths])))
        )


replace_map = {
            'axial': 'mask_isotropic_axial_png',
            'coronal': 'mask_isotropic_coronal_png',
            'sagittal':'mask_isotropic_sagittal_png'
        }

print('total images', len(paths))
print(paths[:3])

# calc tumor class distribution

df_meta = pd.DataFrame({
    'slice_distribution': [p.split(os.sep)[-1][0] for p in paths]
    }
    )
print(df_meta.value_counts())
df_meta = pd.DataFrame(
        {
            'patient_distribution': [pid[0] for pid in np.unique([p.split(os.sep)[-1][:3] for p in paths])]

        }
    )
print(df_meta.value_counts())


paths_mask = [format_mask_path(p) for p in paths]

### DATALOADER ###

train_transforms = A.Compose([
        # A.PadIfNeeded(min_height=args.input_size, min_width=args.input_size, p=1, border_mode=cv2.BORDER_CONSTANT, value = 0),
        # A.RandomCrop(height=args.input_size, width=args.input_size, p=1.0), #randcrop from an ROI
        A.Resize(height=args.input_size, width=args.input_size, interpolation=1, always_apply=True, p=1),
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(gamma_limit=(50, 200), p = 0.5),
        A.RandomBrightnessContrast(p=0.5,),
        A.GaussianBlur(p=0.5),
        A.MotionBlur(p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.Normalize(
            mean = 0,
            std = 1.0,
            max_pixel_value= 255.0
        ),
        ToTensorV2()
])

test_transforms = A.Compose([
        A.Resize(height=args.input_size, width=args.input_size, interpolation=1, always_apply=True, p=1), #randcrop from an ROI
        A.Normalize(
            mean = 0,
            std = 1.0,
            max_pixel_value= 1.0
        ),
        ToTensorV2()
])

### MODEL ####
def build_model(backbone, num_classes, device='cpu'):
    model = smp.Unet(
        encoder_name=backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to(device)
    return model

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for i_fold, item in enumerate(kfold.split(df_consensus)):
    print('==========================================')
    df_train = df_consensus.iloc[item[0]]
    df_valid = df_consensus.iloc[item[1]]
    
    paths_img_train = [p for p in paths if p.split(os.sep)[-1][:3] in df_train.PID.to_list()]
    paths_mask_train = [p for p in paths_mask if p.split(os.sep)[-1][:3] in df_train.PID.to_list()]
    print('train_distributions, fold', i_fold)
    df_meta = pd.DataFrame({
        'slice_distribution': [p.split(os.sep)[-1][0] for p in paths_img_train]
        }
        )
    print(df_meta.value_counts())
    df_meta = pd.DataFrame(
            {
                'patient_distribution': [pid[0] for pid in np.unique([p.split(os.sep)[-1][:3] for p in paths_img_train])]

            }
        )
    print(df_meta.value_counts())

    df_meta = pd.DataFrame(
            {
                'view_distribution': [p.split(os.sep)[-3].split('_')[4] for p in paths_img_train]

            }
        )
    print(df_meta.value_counts())



    dataset_train = MyxoidDataset_T1_from_T2_path(paths_img_train, paths_mask_train, transform = train_transforms)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size= args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
        )


    print('fold: {}, train images: {}'.format(i_fold,len(dataset_train)))
    sam_model = build_medsam_lite('lite_medsam.pth')
    model = MedSAM_Lite_6ch(
        sam_model.image_encoder,
        sam_model.mask_decoder, 
        sam_model.prompt_encoder,
        args.train_mode
    )

    if args.resume_from is not None:
        print('loading model weights from', args.resume_from)
        model.load_state_dict(torch.load(args.resume_from))

    #### TRAIN ###
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    if args.loss_type == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    if args.loss_type == 'bce-dice':
        criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader_train)

    model = model
    model.train()
    losses = []
    print('train...')
    for epoch in range(args.epoch):
        running_loss = 0
        model.train()
        for i, (img, mask) in enumerate(tqdm(dataloader_train)):
            img = img
            mask = mask.unsqueeze(1).float()
            B,_, H, W = mask.shape
            # set the bbox as the image size for fully automatic segmentation
            boxes = torch.from_numpy(np.array([[0,0,W,H]]*B)).float()
            output = model(img,boxes)
            with accelerator.autocast():
                loss = criterion(output, mask.to(device))
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            running_loss += loss.item()
            loss_avg = running_loss/(i+1)
        losses.append(loss_avg)
        print('{}, train: epoch: {}, loss: {:.4f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            epoch,
            loss_avg
            )
            )
        
        if (epoch+1)%args.save_every==0:
            path_save =os.path.join(dir_output,
                                    '{}_fold-{}_epoch-{}_loss-{:.4f}.ckpt'.format(
                                        TIME_STAMP, i_fold, epoch, loss_avg)) 
            print('saving model at', path_save)
            accelerator.save(accelerator.unwrap_model(model).state_dict(), path_save)

    with open(os.path.join(dir_output,'{}_fold-{}_train_loss.txt'.format(i_fold,TIME_STAMP)), 'w') as f:
        for l in losses:
            f.write('{:8f}\n'.format(l))
print(TIME_STAMP)
print('filename', __file__)
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))
 
