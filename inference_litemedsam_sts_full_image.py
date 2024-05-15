#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse


# In[2]:


import sys
import random
import nibabel as nib
from glob import glob
from tqdm.auto import tqdm
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
# import lightly
#from backboned_unet import Unet
# import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from logger import Logger
from utils import MyxoidDataset_T1_from_T2_path, MyxoidDataset_T1_from_T2_path_only_img, dice_coef_multiclass, seed_everything, merge_score
from utils import drawContourMontage, drawMontage, drawContour, soft_dice_score, inference_patch_img_2_5d
import utils
import pandas as pd
import pickle
import cv2
#import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import torchvision.transforms.functional as TF

from sam_surgery import MedSAM_Lite_6ch, build_medsam_lite
from segment_anything import sam_model_registry
from accelerate import load_checkpoint_and_dispatch
from accelerate import Accelerator
# In[3]:


parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type = float, default = 0.25)
parser.add_argument('--gpus', default = '0')
parser.add_argument('--fold', type = int, default = 0)
parser.add_argument('--epoch', type = int, default = 4)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--input_size', type = int, default = 256)
parser.add_argument('--p_thresh', type = float, default = 0.0)
parser.add_argument('--num_workers', type = int, default = 4)
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--tta', type = int, default = 1)
parser.add_argument('--multiview', type = int, default = 1)
parser.add_argument('--timestamp', default = '2024-03-26-02-30-27')
parser.add_argument('--img_type', type = str, default='T2')
parser.add_argument('--output_merge_mode', type = str, default= 'avg')

# '/home/reasatt/Projects/napari_annotation/model_data/2023-01-13-21-55-21/epoch=499-step=390500-train_loss_ssl=3.19099450.ckpt'
args = parser.parse_args()
for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))
 

# In[73]:


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' #or CUBLAS_WORKSPACE_CONFIG=:16:8


# In[4]:


def format_mask_path(p):
    dir = p.split(os.sep)[-3]
    if 'axial' in dir:
        mask_dir = replace_map['axial']
    if 'coronal' in dir:
        mask_dir = replace_map['coronal']
    if 'sagittal' in dir:
        mask_dir = replace_map['sagittal']

    p_mask = p.replace(dir, mask_dir)
    assert os.path.exists(p_mask), (p, p_mask)
    return p_mask


# In[5]:


# pids_filtered = [
#    'L44','F02','N21'
# ]


# In[6]:


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

replace_map = {
            'axial': 'mask_isotropic_axial_png',
            'coronal': 'mask_isotropic_coronal_png',
            'sagittal':'mask_isotropic_sagittal_png'
        }


# In[7]:


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


# In[8]:


def normalize_mri(volume):
    lower = np.percentile(volume.flatten(),q=0.5)
    upper = np.percentile(volume.flatten(),q=99.5)
    volume = np.clip(a = volume, a_min=lower, a_max=upper)
    volume = volume.astype(np.float32)
    volume = (volume-np.min(volume))/(np.max(volume)-np.min(volume))
    volume = 255.0*volume
    return volume


from operator import itemgetter

def create_output_volume(pid,df_label, 
                         outputs_slices,
                        view_list = ['axial','coronal', 'sagittal']
                        ):
    scores_patient = {}
    outputs_volume_patient_views = []
    for view in view_list:
        df_patient = df_label.groupby(['PID','view']).get_group((pid,view)).sort_values('flnames')
        
        indices = df_patient.index.values
        outputs_volume_patient = itemgetter(*indices)(outputs_slices)
        outputs_volume_patient = np.array(outputs_volume_patient)
        if view == 'axial':
            # print('axial', outputs_volume_patient.shape)
            outputs_volume_patient_views.append(outputs_volume_patient)
        if view == 'coronal':
            # print('coronal', outputs_volume_patient.shape)
            outputs_volume_patient_views.append(np.transpose(outputs_volume_patient,(2,0,1)))
        if view == 'sagittal':
            # print('sagittal', outputs_volume_patient.shape)
            outputs_volume_patient_views.append(np.transpose(outputs_volume_patient,(2,1,0)))
    
#     if len(view_list)>1:
#         assert outputs_volume_patient_views[0].shape == outputs_volume_patient_views[1].shape == outputs_volume_patient_views[2].shape, '{} {} {} {}'.format(pid,outputs_volume_patient_views[0].shape, outputs_volume_patient_views[1].shape, outputs_volume_patient_views[2].shape)
    
    return outputs_volume_patient_views

def merge_multiview(outputs_volume_patient_views, thresh = 0.0, axis_weights = None):
#     assert outputs_volume_patient_views.shape == 4 # first dim is views
    if axis_weights is None:
        axis_weights = np.ones(len(outputs_volume_patient_views))/len(outputs_volume_patient_views)
        # equal weights
    pred_vol = np.zeros_like(outputs_volume_patient_views[0])
    for i,vol in enumerate(outputs_volume_patient_views):
        # print(axis_weights.shape, pred_vol.shape, vol.shape)
        pred_vol+=vol*axis_weights[i]
    pred_vol = ((torch.tensor(pred_vol))>thresh).numpy().astype(np.uint8)
#     print(pred_vol.shape)
    pred_vol = np.transpose(pred_vol, (1,2,0))
    return pred_vol

def drawMontageSingleChannel(img, sample_num = 16, grid_shape = (4,4)):
    '''
    img shape: h,w, slice_num
    '''
    
    range_ind = list(range(0, img.shape[-1]//sample_num*sample_num, img.shape[-1]//sample_num))
#     print(range_ind)
    img_overlay_slices = img[:,:,range_ind].transpose(2,0,1)
    img_mt = montage(img_overlay_slices,grid_shape = grid_shape, multichannel=False)
    return img_mt

def drawContourVolume(main_vol, seg_vol, color = (0,255,0), thickness=1, multichannel = False):
    '''
    # main --> rgb, uint8
    # seg --> gray, unit8
    # Find external contours
    '''
    overlay_img_vol = []
    for i in range(main_vol.shape[-1]):
        if multichannel:
            img_slice = main_vol[:,:,:,i]
        else:
            img_slice = main_vol[:,:,i]
            img_slice = np.tile(img_slice[:,:,None],[1,1,3])
        img_slice = np.ascontiguousarray(img_slice)
        overlay_img_slice = drawContour(img_slice.copy(), seg_vol[:,:,i].copy(), color, thickness)
        overlay_img_vol.append(overlay_img_slice)
 
    overlay_img_vol = np.array(overlay_img_vol)
    return overlay_img_vol.transpose(1,2,3,0)

#tta 
def tta1(x, k):
    if k== 0: return x
    if k== 1: return torch.flip(x, dims=[2,])  #tta
    if k==-1: return torch.flip(x, dims=[2,])  #undo tta
def tta2(x, k):
    if k== 0: return x
    if k== 1: return torch.flip(x, dims=[3,]) 
    if k==-1: return torch.flip(x, dims=[3,]) 
def tta3(x, k):
    if k== 0: return x
    if k== 1: return torch.rot90(x, k= 1, dims=[2,3])
    if k==-1: return torch.rot90(x, k=-1, dims=[2,3])

if args.tta:
    CFG_LIST = [
            [0,0,0],
            [1,0,0],
            [0,1,0],
            [1,1,0],
            [0,0,1],
            [1,0,1],
            [0,1,1],
            [1,1,1], 
        ]
else:
        CFG_LIST = None

# In[9]:

print(args.timestamp)


# In[10]:

test_transforms = A.Compose([
        A.Resize(height=args.input_size, width=args.input_size, interpolation=1, always_apply=True, p=1), #randcrop from an ROI
        A.Normalize(
            mean = 0,
            std = 1.0,
            max_pixel_value= 255.0
        ),
        ToTensorV2()
])


df_consensus = pd.read_csv('/scratch/reasatt/STS_public/image_metadata.csv')
pids_filtered  = []
'''
for pid, group in df_consensus.groupby('PID'):
    shapes = group[['Height', 'Width', 'SliceNum','PixelSpacing','SliceThickness']].to_numpy()
    if np.array_equal(shapes[0], shapes[1]):
        pids_filtered.append(pid)
'''
pids_filtered = df_consensus.PID.unique()
pids_filtered = [pid for pid in pids_filtered if pid not in ['STS_025', 'STS_029']]
print('pids_filtered', len(pids_filtered))
# pids_filtered = pids_filtered[:1]

view_list = ['axial','coronal', 'sagittal']

paths = sorted(glob('/scratch/reasatt/STS_public/image_2_5d_isotropic_*_stride-1/*/*'))
print(len(paths))
paths = [p for p in paths if p.split(os.sep)[-1][:7] in pids_filtered]
print(len(paths))

paths = [p for p in paths if args.img_type in p]

replace_map = {
            'axial': 'mask_axial_png',
            'coronal': 'mask_coronal_png',
            'sagittal':'mask_sagittal_png'
        }

# paths_mask = [format_mask_path(p) for p in paths]
print('total images', len(paths))
# print('total mask paths', len(paths_mask))
print(paths[:3])


############# MODEL ##########################

### MODEL ####
# model = Unet(backbone_name='resnet18',pretrained = True, classes = 1)
# model.backbone.conv1 = torch.nn.Conv2d(6,64,kernel_size=(7,7), stride=(2,2), padding=(3,3), bias= False)


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


df_label = pd.DataFrame(data= {
    'paths' : paths,
    'PID' : [p.split(os.sep)[-2][:7] for p in paths],
    'Modality' : [p.split(os.sep)[-2][8:10] for p in paths],
    'flnames': [p.split(os.sep)[-1] for p in paths]
}
) 
df_label['view'] = df_label.paths.apply(lambda x: x.split(os.sep)[-3].split('_')[4])

model_paths_unsorted = glob('/tank/data/Project-reasatt/soft_tissue_tumor/model_data/{}/*.ckpt'.format(args.timestamp))
print(model_paths_unsorted)
epoch_num = [int(p.split('_')[-2].replace('epoch-','')) for p in model_paths_unsorted]
model_paths = np.array(model_paths_unsorted)[np.argsort(epoch_num)]
df_model_path = pd.DataFrame({
    'path': model_paths,
    'fold': [p.split(os.sep)[-1].split('_')[1] for p in model_paths],
    'epoch': [p.split(os.sep)[-1].split('_')[2] for p in model_paths]
})

model_path = df_model_path.groupby(['fold', 'epoch']).get_group(
    ('fold-{}'.format(args.fold), 'epoch-{}'.format(args.epoch))
).path.item()

'''
state_dict = torch.load(model_path)
for key in list(state_dict.keys()):
    state_dict[key.replace('module.', '')] = state_dict.pop(key)    
model.load_state_dict(state_dict)
'''
sam_model = build_medsam_lite('lite_medsam.pth')
model = MedSAM_Lite_6ch(
    sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder,
    train_mode = 'finetune-decoder'
)


#accelerator = Accelerator()
#model = accelerator.prepare(model)
model.load_state_dict(torch.load(model_path))

model = model.cuda()
_=model.eval()

outputs_slices_all = []
df_valid_reconstructed = []
for (pid, v), df_group in df_label.groupby(['PID','view']):
    # if pid != 'STS_022':
    #     continue
    print('========================')
    print(pid,v)
    print('========================')
    dataset = MyxoidDataset_T1_from_T2_path_only_img(
        paths_img = df_group.paths.values, 
        paths_mask = None,
        transform = test_transforms
    )

    print('slices', len(dataset))
    h,w,_ = np.array(Image.open(dataset.paths_img[0])).shape
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size = args.batch_size, 
            shuffle = False,
            num_workers = args.num_workers,
            pin_memory = True
        )
    ############### INFERENCE ###############################
    print('inference iterating slice batches','slice shape', h,w)
    outputs_slices = []
    for img_batch, _ in dataloader:
        print('slices batch shape', img_batch.shape) 
        with torch.no_grad():
            B,_, H_b, W_b = img_batch.shape
            # set the bbox as the image size for fully automatic segmentation
            boxes = torch.from_numpy(np.array([[0,0,W_b,H_b]]*B)).float().cuda()
            outputs = []
            for t1,t2,t3 in CFG_LIST:
                out = model(tta1(tta2(tta3(img_batch.cuda(),t3),t2),t1),boxes)
                out = out.detach().cpu()
                out = tta3(tta2(tta1(out,-t1),-t2),-t3)
                
                out = TF.resize(out, (h, w), antialias=True)
                
                outputs.append(out)
                
                
            outputs = torch.stack(outputs)
            
            # print(outputs.shape)
            outputs = outputs.mean(dim = 0)
            # outputs = torch.sigmoid(outputs)

            # print(outputs.shape)
            outputs_slices.append(outputs.squeeze(1))
        
    outputs_slices = np.concatenate(outputs_slices) # slice, h, w
    print('outputs_slices', outputs_slices.shape)
    #print(np.max(outputs_slices))
    outputs_slices_all.append(outputs_slices) # pid, slice, h, w
    df_valid_reconstructed.append(df_group)
outputs_slices_all = [item2 for item1 in outputs_slices_all for item2 in item1]

print('outputs_slices_all', len(outputs_slices_all))
df_valid = pd.concat(df_valid_reconstructed).reset_index(drop=True)

print('df_valid constructed')
print(df_valid.head())
print(len(df_valid))


dice_list = []

for PID in tqdm(sorted(df_valid.PID.unique())):
#     if row['PID'] not in pt_select:
#         continue
    print('creating prediction volume ',PID)

    outputs_volume_patient_views = create_output_volume(
        PID,
        df_valid,
        outputs_slices_all,
        view_list
        )
    for i, vol in enumerate(outputs_volume_patient_views):
        print(PID, i, vol.shape)
    pred_vol = merge_multiview(outputs_volume_patient_views,thresh = args.p_thresh)
    print('pred_vol', pred_vol.shape)
    print('computing dice ...')
    path_target = os.path.join(
            '/scratch/reasatt/STS_public/mask_isotropic_nii','{}_{}.nii.gz'.format(
                PID,
                'T2'
            ))
    
    target_vol = nib.load(path_target).get_fdata()
    print('target_vol', target_vol.shape)
    d = soft_dice_score(torch.tensor(pred_vol), torch.tensor(target_vol))
#     print(d)
    dice_list.append(d.item())

df_dice = pd.DataFrame(
    {
        'PID': sorted(df_valid.PID.unique()),
        'dice': dice_list,
    }
)
df_dice = df_dice.sort_values(by = 'dice')
print(df_dice.dice.to_list())
print(df_dice.dice.mean())

df_dice.to_csv(os.path.join('/tank/data/Project-reasatt/soft_tissue_tumor/model_data', args.timestamp,
'sts_full_image_metrics_fold-{}_epoch-{}_ps-{}_p-thresh-{}_tta-{}.csv'.format(
    args.fold,
    args.epoch,
    args.input_size,
    args.p_thresh,
    args.tta
    )),
index = False)
print(args.timestamp)

for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))

