import numpy as np
import cv2
from skimage.util import montage
import torch
from PIL import Image
# import pytorch_lightning as pl
import random
import nibabel as nib
from glob import glob
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import numpy as np
import time
from PIL import Image
import os
import pandas as pd
from empatches import EMPatches 
import pytorch_lightning as pl
import albumentations as A
import torchvision.transforms.functional as TF

def seed_everything(seed = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)

def drawContour(main, seg, color = (0,255,0), thickness=1):
    '''
    # main --> rgb, uint8
    # seg --> gray, unit8
    # Find external contours
    '''
    contours,_ = cv2.findContours(seg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # Iterate over all contours
    for i,c in enumerate(contours):
        cv2.drawContours(main,[c],-1,color,thickness)
    return main
    
    
def drawMask(main, seg, color = (0,255,0), alpha=0.2):
    
    masked_img = np.where(seg[...,None], color, main).astype(np.uint8)
    out = cv2.addWeighted(main, 1-alpha, masked_img, alpha,0)    
    
    return out


def drawContourMontage(img, mask, sample_num = 16, grid_shape = (4,4)):
    '''
    img shape: h,w, slice_num
    '''
    range_ind = list(range(0, img.shape[-1]//sample_num*sample_num, img.shape[-1]//sample_num))
#     print(range_ind)
    img_overlay_slices = []
    for i in range_ind:
        seg =(mask[:,:,i]*255).astype(np.uint8)
        main = cv2.cvtColor(img[:,:,i].astype(np.uint8),cv2.COLOR_GRAY2RGB)
        if seg.sum()==0:
            img_overlay = main
        else:
            img_overlay = drawContour(main, seg)
        img_overlay_slices.append(img_overlay)
    img_mt = montage(img_overlay_slices,grid_shape = grid_shape, channel_axis=-1)
    return img_mt
   
def drawMontage(img, sample_num = 16, grid_shape = (4,4)):
    '''
    img shape: h,w,c, slice_num
    '''
    
    range_ind = list(range(0, img.shape[-1]//sample_num*sample_num, img.shape[-1]//sample_num))
#     print(range_ind)
    img_overlay_slices = img[:,:,:,range_ind].transpose(3,0,1,2)
    img_mt = montage(img_overlay_slices,grid_shape = grid_shape, channel_axis=-1)
    return img_mt

def padSlice(values, c_x, c_y):

    target_shape = np.array((c_x, c_y))
    pad = ((target_shape - values.shape) / 2).astype("int")

    values = np.pad(values, ((pad[0], pad[0]), (pad[1], pad[1])), mode="constant", constant_values = 0)

    return values

def formatSignal(path_w, c_x=512, c_y=512, c_crop_slices=0):

    data = nib.load(path_w)
    img_w = data.get_fdata()

    slices_out = []

    Z = img_w.shape[2]

    # For each axial slice
    for z in range(Z):

        # Skip some top and bottom slices due to folding artefacts
        if z < c_crop_slices or (img_w.shape[2] - z) <= c_crop_slices: continue

        # Get coordinates of three adjacent slices, periodic border condition
        z_range = np.clip(np.arange(-1,2)+z, c_crop_slices, Z-c_crop_slices-1)
        print(z_range)

        # Initialize sample, three adjacent axial slices
        slice_img = np.zeros((c_x, c_y, 3))

        # Extract slices with pre-processing
        for i in range(3):
            z_r = z_range[i]

            slice_w = img_w[:, :, z_r]
#             slice_w = normalizeClip(slice_w)
            slice_w = padSlice(slice_w)

            slice_img[:, :, i] = slice_w

        # Change indexing order:
        # (x, y, c) to (c, y, x)
        slice_img = np.swapaxes(slice_img, 0, 2)
        slice_img = np.reshape(slice_img, (3, c_y, c_x))

        slices_out.append(slice_img)
    
    slices_out = np.array(slices_out)

    return (slices_out, img_w.shape)


class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, norm_stat):
        super().__init__()
        self.imgs = imgs
        self.norm_stat = norm_stat

    def __len__(self):
        return(len(self.imgs))
    def __getitem__(self, idx):
        imgs = self.imgs[idx]
        imgs = imgs/255.0
        imgs = (imgs - self.norm_stat['mean'])/self.norm_stat['std']
        imgs = torch.tensor(imgs)
        imgs = imgs.permute(2,0,1)
        #print(imgs.min(),  imgs.max())
        return imgs

class OutputMergeDataset(torch.utils.data.Dataset):
    def __init__(self, patches_groups, H, W, emp, indices):
        super().__init__()
        self.patches_groups = patches_groups
        self.H =H
        self.W = W
        self.emp = emp
        self.indices = indices

    def __len__(self):
        return(len(self.patches_groups))
    def __getitem__(self, idx):
        out = self.patches_groups[idx]
        outputs_tile = self.emp.merge_patches(out.permute(0,2,3,1),self.indices, mode = 'avg')
        outputs_tile = outputs_tile.squeeze()[:self.H,:self.W] # exclude padding
        return outputs_tile

class STT_View(torch.utils.data.Dataset):
    def __init__(self, img_paths):
        super().__init__()
        # self.img = img
        # print(self.img.shape)
        self.img_paths = None

    def __len__(self):
        return(len(self.img_paths))

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = np.array(Image.open(path))
        return img

class MyxoidDataset(torch.utils.data.Dataset):
    def __init__(self, paths_img, paths_mask, transform=None):
        super().__init__()
        self.paths_img = paths_img
        self.paths_mask = paths_mask
        self.transform = transform
    def __len__(self):
        return(len(self.paths_img))
    def __getitem__(self, idx):
        p_img = self.paths_img[idx]
        p_mask = self.paths_mask[idx]
        img = np.array(Image.open(p_img))
        mask = np.array(Image.open(p_mask))
        if self.transform is not None:
            aug = self.transform(image = img, mask = mask)
            img, mask = aug['image'], aug['mask']
        return img, mask
        
class MyxoidDataset_T1_from_T2_path(MyxoidDataset):
    def do_transform(self, img, mask):
        aug = self.transform(image = img, mask = mask)
        img, mask = aug['image'], aug['mask']
        return img, mask
        
    def __getitem__(self, idx):
        p_img_t2 = self.paths_img[idx]
        p_mask = self.paths_mask[idx]
        img_t2 = np.array(Image.open(p_img_t2))
        p_img_t1 = p_img_t2.replace('T2','T1')#.replace('image','image_reg_T1')
        img_t1 = np.array(Image.open(p_img_t1))
        mask = np.array(Image.open(p_mask))
        img = np.concatenate((img_t1, img_t2), axis = 2)
        if self.transform is not None:
            img, mask = self.do_transform(img, mask)
        return img, mask
class MyxoidDataset_T1_from_T2_path_single_channel(MyxoidDataset):
    def do_transform(self, img, mask):
        aug = self.transform(image = img, mask = mask)
        img, mask = aug['image'], aug['mask']
        return img, mask
        
    def __getitem__(self, idx):
        p_img_t2 = self.paths_img[idx]
        p_mask = self.paths_mask[idx]
        img_t2 = np.array(Image.open(p_img_t2))
        p_img_t1 = p_img_t2.replace('T2','T1')#.replace('image','image_reg_T1')
        img_t1 = np.array(Image.open(p_img_t1))
        mask = np.array(Image.open(p_mask))
        img = np.concatenate((img_t1, img_t2), axis = 2)
        img = img[:,:,[1,4]]
        if self.transform is not None:
            img, mask = self.do_transform(img, mask)
        return img, mask
     
class MyxoidDataset_T1_from_T2_path_only_img(MyxoidDataset):
    def __getitem__(self, idx):
        p_img_t2 = self.paths_img[idx]
#         p_mask = self.paths_mask[idx]
        img_t2 = np.array(Image.open(p_img_t2))
        p_img_t1 = p_img_t2.replace('T2','T1')
        assert p_img_t2 != p_img_t1 
#         p_img_t1 = p_img_t2.replace('image','image_reg_T1')

        img_t1 = np.array(Image.open(p_img_t1))
#         mask = np.array(Image.open(p_mask))
        img = np.concatenate((img_t1, img_t2), axis = 2)
        mask = np.zeros_like(img)
        if self.transform is not None:
            aug = self.transform(image = img, mask = mask)
            img, mask = aug['image'], aug['mask']
        return img, mask

class MyxoidDataset_T1_from_T2_path_only_img_single_channel(MyxoidDataset):
    def __getitem__(self, idx):
        p_img_t2 = self.paths_img[idx]
#         p_mask = self.paths_mask[idx]
        img_t2 = np.array(Image.open(p_img_t2))
        p_img_t1 = p_img_t2.replace('T2','T1')
        assert p_img_t2 != p_img_t1 
#         p_img_t1 = p_img_t2.replace('image','image_reg_T1')

        img_t1 = np.array(Image.open(p_img_t1))
#         mask = np.array(Image.open(p_mask))
        img = np.concatenate((img_t1, img_t2), axis = 2)
        img = img[:,:,[1,4]]
        mask = np.zeros_like(img)
        if self.transform is not None:
            aug = self.transform(image = img, mask = mask)
            img, mask = aug['image'], aug['mask']
        return img, mask
class MyxoidDataset_T2_only_img(MyxoidDataset):
    def __getitem__(self, idx):
        p_img_t2 = self.paths_img[idx]
#         p_mask = self.paths_mask[idx]
        img = np.array(Image.open(p_img_t2))
#         mask = np.array(Image.open(p_mask))
        mask = np.zeros_like(img)
        if self.transform is not None:
            aug = self.transform(image = img, mask = mask)
            img, mask = aug['image'], aug['mask']
        return img, mask
  

def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.00001,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def dice_coef_multiclass(y_true, y_pred):
    # img, cls, h, w
    num_imgs, numLabels = y_true.shape[0], y_true.shape[1]
    dice_img_cls = torch.zeros(
        (num_imgs, numLabels)
    )
    for index in range(numLabels):
        dice_imgs = soft_dice_score(
            y_true[:,index,:,:], y_pred[:,index,:,:],
            dims = [1,2]
        )
        dice_img_cls[:,index] = dice_imgs
    return dice_img_cls # taking average


def inference_patch_img_2_5d(
        model,
    
        dataset,
        ind,
        win,
        pad_value,
        norm_stat={'mean': 0.0, 'std': 1.0},
        overlap = 0,
        args = None,
        arch_type = 'unet'
):
    emp = EMPatches()
    model.eval()
    model = model.cuda()
    dice_scores_img_cls = []
#     for i, (img, mask) in enumerate(dataset):
        # extract image patch 
    img, mask = dataset[ind]
    img = np.array(img.squeeze())
#     print(img.shape)
#     print(dataset.paths_img[ind])
#     print(img.min(), img.max(), img.mean())

#     print('img.shape',img.shape)
    h,w = img.shape[0], img.shape[1]
    pad = ((win-h%win)%win, (win-w%win)%win)
#     print(pad)

    img_padded = cv2.copyMakeBorder(
        img, 
        top = 0,
        bottom=pad[0],
        left=0,
        right = pad[1],
        borderType=cv2.BORDER_CONSTANT, 
        value = pad_value
    )
#     print(img_padded.min(), img_padded.max(), img_padded.mean())
    img_padded = img_padded.astype(np.uint8)
#     print(img_padded.min(), img_padded.max(), img_padded.mean())

    #print('img_padded.shape', img_padded.shape)
    img_patches, indices = emp.extract_patches(img_padded, patchsize=win, overlap=overlap)

    img_patches = np.array(img_patches)
    #print('img_patches.shape', img_patches.shape)

    # do norm and to tensor
    img_patches = img_patches.astype(np.float32)/255.0
    img_patches = (img_patches - norm_stat['mean'])/norm_stat['std']
    #print(img_patches.min(), img_patches.max())
    img_patches = torch.tensor(img_patches)
    img_patches = img_patches.permute(0,3,1,2)
    #print('img_patches.shape', img_patches.shape)

    # compute pred masks for patch
    with torch.no_grad():
        outputs_all = []
        for ind in range(0,img_patches.shape[0],args.batch_size):
            img_batch = img_patches[ind: ind+args.batch_size].cuda()
#             print(img_batch.min().item(), img_batch.max().item(), img_batch.mean().item())
            if arch_type == 'unet':
                outputs = model(img_batch)
            if arch_type == 'sam':
                B,_, H, W = img_batch.shape
                # set the bbox as the image size for fully automatic segmentation
                boxes = torch.from_numpy(np.array([[0,0,W,H]]*B)).float().cuda()
                img = TF.resize(img, (1024, 1024), antialias=True)
                outputs = model(img_batch, boxes)
            outputs_all.append(outputs.detach().cpu())
        outputs_all = torch.cat(outputs_all)
        #print('outputs_all.shape',outputs_all.shape)
    
    # stitch output preds to pred mask
    outputs_tile = emp.merge_patches(outputs_all.permute(0,2,3,1),indices, mode = 'avg')
#     print(outputs_tile.shape)
    outputs_tile = outputs_tile.squeeze()[:h,:w] # exclude padding
#     outputs_tile = torch.tensor(outputs_tile).permute(2,0,1)
#     print('outputs_tile.shape', outputs_tile.shape)
    
    return outputs_tile



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

def split_patch_infer_merge(
        model,
        img_list,
        win,
        pad_value,
        norm_stat={'mean': 0.0, 'std': 1.0},
        args = None,
	tta_list = None,
        c = 6,
        arch_type = 'unet',
        mp = None,
        engine = None
):
    
    """
    image list come from the sme volume so that hand w is consisitent
    """
    print('processing slice to patches...')
    start = time.time()
    emp = EMPatches()
    img_patches_list = []
    indices_list = []

    img = img_list[0]
    H,W = img.shape[0], img.shape[1]
    pad = ((win-H%win)%win, (win-W%win)%win)

    # tx = A.PadIfNeeded(
    #         min_height=H+pad[0],
    #         min_width=W+pad[1], 
    #         p=1,
    #         border_mode=cv2.BORDER_CONSTANT, 
    #         value = pad_value)
    # img_padded = tx(image=img)['image']

    img_padded = cv2.copyMakeBorder(
        img, 
        top = 0,
        bottom=pad[0],
        left=0,
        right = pad[1],
        borderType=cv2.BORDER_CONSTANT, 
        value = pad_value
    )
     
    img_padded = img_padded.astype(np.uint8)
#     print(img_padded.min(), img_padded.max(), img_padded.mean())

    #print('img_padded.shape', img_padded.shape)
    img_patches, indices = emp.extract_patches(img_padded, patchsize=win, overlap=args.patch_ol)
    patch_num = len(img_patches)


    img_patches_list = np.zeros(
            (len(img_list), patch_num, win,win,c),
            dtype = np.uint8
            )
     
    for i_img, img in enumerate(img_list):
        img_padded = cv2.copyMakeBorder(
            img, 
            top = 0,
            bottom=pad[0],
            left=0,
            right = pad[1],
            borderType=cv2.BORDER_CONSTANT, 
            value = pad_value
        )
        # img_padded = tx(image=img)['image']
        # print(img_padded.min(), img_padded.max(), img_padded.mean())
        img_padded = img_padded.astype(np.uint8)
        # print('img_padded',img_padded.min(), img_padded.max(), img_padded.mean())

        #print('img_padded.shape', img_padded.shape)
        img_patches, indices = emp.extract_patches(img_padded, patchsize=win, overlap=args.patch_ol)
        
        indices_list.append(indices)
        img_patches = np.array(img_patches)
        # print('img_patches', img_patches.shape)
        img_patches_list[i_img,:,:,:,:]= img_patches
    
    print('reshaping patches...')
    
    print('img_patches_list.shape',img_patches_list.shape) # img_list_num, patch_num, H, W, C
    img_patches_list = img_patches_list.reshape(len(img_list)*patch_num, win,win,c)
    print(img_patches_list.shape)

    
    print('setting up patch dataloder')
    dataset = Image_Dataset(img_patches_list, norm_stat)
    # print(dataset[0].max(), dataset[0].min())
    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    if mp == 'ddp':
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size= args.batch_size_patch, 
            shuffle=False, #train_sampler is None),
            num_workers=args.num_workers, 
            pin_memory=True,
            sampler=train_sampler,
            # multiprocessing_context='spawn'
            )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size = args.batch_size_patch,
            shuffle =  False, 
            num_workers = args.num_workers
        )    
    model.eval()
    # compute pred masks for patch
    print('computing model output ...')
    if engine:
        model, dataloader = engine.prepare(model, dataloader)
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                outputs_all = []
                if tta_list:
                    for i, img_batch in enumerate(dataloader):
                        # print(i,'img_batch',img_batch.shape)
                        outputs = []
                        img_batch = img_batch.half()
                        # print('img stat', img_batch.shape, img_batch.max().item(), img_batch.min().item())
                        #        img_batch.min().item(), img_batch.sum().item())
                        for t1,t2,t3 in tta_list:
                            if arch_type == 'unet':
                                out = model(tta1(tta2(tta3(img_batch,t3),t2),t1))
                            if arch_type == 'sam':
                                B,_, H_b, W_b = img_batch.shape
                                # set the bbox as the image size for fully automatic segmentation
                                boxes = torch.from_numpy(np.array([[0,0,W_b,H_b]]*B)).float()
                                img_batch = TF.resize(img_batch, (1024, 1024), antialias=True)
                                out = model(tta1(tta2(tta3(img_batch,t3),t2),t1), boxes)
                                out = engine.gather(out)
                            if arch_type == 'litesam':
                                B,_, H_b, W_b = img_batch.shape
                                # set the bbox as the image size for fully automatic segmentation
                                boxes = torch.from_numpy(np.array([[0,0,W_b,H_b]]*B)).float()
                                # img_batch = TF.resize(img_batch, (1024, 1024), antialias=True)
                                out = model(tta1(tta2(tta3(img_batch,t3),t2),t1), boxes)
                                out = engine.gather(out)
                        
                            # print('out stat', out.max().item(), out.min().item())
                            out = out.detach().cpu()
                            out = tta3(tta2(tta1(out,-t1),-t2),-t3)
                            outputs.append(out)
                        outputs = torch.stack(outputs)
                        # print(outputs.shape)
                        outputs = outputs.sum(dim = 0)
                        outputs = outputs/len(tta_list)
                        outputs = torch.sigmoid(outputs)
                        # print('output stat', outputs.max().item(), outputs.min().item())
                        outputs_all.append(outputs.detach().cpu())
    else:
        
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                outputs_all = []
                if tta_list:
                    for i, img_batch in enumerate(dataloader):
                        print(i,'img_batch',img_batch.shape)
                        outputs = torch.zeros((img_batch.shape[0],1,img_batch.shape[2],img_batch.shape[3]))
                        img_batch = img_batch.cuda().half()
                        # print('img stat', img_batch.shape, img_batch.max().item(), img_batch.min().item())
                        #        img_batch.min().item(), img_batch.sum().item())
                        for t1,t2,t3 in tta_list:
                            if arch_type == 'unet':
                                out = model(tta1(tta2(tta3(img_batch,t3),t2),t1))
                            if arch_type == 'sam':
                                B,_, H_b, W_b = img_batch.shape
                                # set the bbox as the image size for fully automatic segmentation
                                boxes = torch.from_numpy(np.array([[0,0,W_b,H_b]]*B)).float().cuda()
                                img_batch = TF.resize(img_batch, (1024, 1024), antialias=True)
                                out = model(tta1(tta2(tta3(img_batch,t3),t2),t1), boxes)
                        
                            # print('out stat', out.max().item(), out.min().item())
                            out = out.detach().cpu()
                            outputs += tta3(tta2(tta1(out,-t1),-t2),-t3)
                        
                        outputs = outputs/len(tta_list)
                        outputs = torch.sigmoid(outputs)
                        # print('output stat', outputs.max().item(), outputs.min().item())
                        outputs_all.append(outputs.detach().cpu())
                else:
                    for img_batch in dataloader:
                        #print('img stat', img_batch.shape, img_batch.max().item(),
                        #        img_batch.min().item(), img_batch.sum().item())
                        outputs = model(img_batch.cuda().float())
                        outputs = torch.sigmoid(outputs)
                        outputs_all.append(outputs.detach().cpu())
    outputs_all = torch.cat(outputs_all)
                          
    # print('outputs_all.shape',outputs_all.shape)
    total_patch_num = len(img_list)*patch_num
    if outputs_all.shape[0]>total_patch_num:
        # print('last batch, truncating padded data')
        outputs_all = outputs_all[:total_patch_num,...]
        # print('outputs_all.shape',outputs_all.shape)
    outputs_all = outputs_all.reshape(len(img_list), patch_num, 1, win,win)
    
    # stitch output preds to pred mask
    print('merging output patches...')
    dataset = OutputMergeDataset(outputs_all, H, W, emp, indices)

    batch_size = 8
    dataloader = torch.utils.data.DataLoader(dataset, shuffle = False,
            batch_size = batch_size, pin_memory = False, num_workers=0)
    
    outputs_tile_array = np.zeros((len(img_list), H,W))
    for i, batch in enumerate(dataloader):
        outputs_tile_array[i*batch_size:i*batch_size+batch_size,:,:] = batch.numpy()
        
        # print('outputs_tile_array stat', outputs_tile_array.max().item(), outputs_tile_array.min().item())
#     print(outputs_tile.shape)
#     outputs_tile = torch.tensor(outputs_tile).permute(2,0,1)
#     print('outputs_tile.shape', outputs_tile.shape)
    
    return outputs_tile_array
def merge_score(outputs_volume_patient_views, mode, mask_volume_axial):
    if mode == 'avg':
        out_vol_merge = np.zeros_like(outputs_volume_patient_views[0]) # axial
        for out in outputs_volume_patient_views:
            out_vol_merge+=out
        out_vol_merge = out_vol_merge/3
        score = soft_dice_score(
                torch.sigmoid(torch.tensor(out_vol_merge))>0.5,
                torch.tensor(mask_volume_axial)
                )# score: pid_merge-view
    if mode == 'max':
        out_vol_merge = np.array(outputs_volume_patient_views).max(axis = 0)
        score = soft_dice_score(
                torch.sigmoid(torch.tensor(out_vol_merge))>0.5,
                torch.tensor(mask_volume_axial)
                )# score: pid_merge-view


    if mode == 'min':
        out_vol_merge = np.array(outputs_volume_patient_views).min(axis = 0)
        score = soft_dice_score(
                torch.sigmoid(torch.tensor(out_vol_merge))>0.5,
                torch.tensor(mask_volume_axial)
                )# score: pid_merge-view


    if mode == 'majority':
        pred_views = torch.sigmoid(torch.tensor(np.array(outputs_volume_patient_views)))>0.5 
        pred_views = pred_views.sum(dim=0)>2

        score = soft_dice_score(
                pred_views,
                torch.tensor(mask_volume_axial)
                )# score: pid_merge-view


    if mode == 'any':
        pred_views = torch.sigmoid(torch.tensor(np.array(outputs_volume_patient_views)))>0.5 
        pred_views = pred_views.sum(dim=0)>0

        score = soft_dice_score(
                pred_views,
                torch.tensor(mask_volume_axial)
                )# score: pid_merge-view
    return score.item()

