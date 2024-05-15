from segment_anything import sam_model_registry
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import collections
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT

class MedSAM(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.channel_downsample = torch.nn.Conv2d(6,3,
                                                  kernel_size=(7,7),
                                                  stride=(1,1),
                                                  padding=(3,3),
                                                  bias= False
                                                 )
        self.image_encoder = torch.nn.Sequential(
            collections.OrderedDict([
                ('channel_downsample', self.channel_downsample),
                ('image_encoder', image_encoder)
            ]
        )
        )
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, boxes):
        img_down = self.image_encoder.channel_downsample(image)  # (B,256,64,64)
        with torch.no_grad():
            image_embeddings = self.image_encoder.image_encoder(img_down)
        
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=boxes[:, None, :],
                masks=None,
            )
            
        # predicted masks
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          )
        # ori_res_masks = F.interpolate(
        #     low_res_masks,
        #     size=(image.shape[2], image.shape[3]),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        return low_res_masks

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder,
        ):
        super().__init__()
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embeddings = self.image_encoder(img_down)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

class MedSAM_Lite_6ch(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder,
            train_mode # finetune-decoder, finetune-encoder-decoder
        ):
        super().__init__()
        self.channel_downsample = torch.nn.Conv2d(6,3,
                                                  kernel_size=(7,7),
                                                  stride=(1,1),
                                                  padding=(3,3),
                                                  bias= False
                                                 )
        self.image_encoder = torch.nn.Sequential(
            collections.OrderedDict([
                ('channel_downsample', self.channel_downsample),
                ('image_encoder', image_encoder)
            ]
        )
        )
        # self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.train_mode = train_mode

    def forward(self, image, box_np):
        if self.training:
            assert self.train_mode in ['finetune-decoder', 'finetune-encoder-decoder']

        img_down = self.image_encoder.channel_downsample(image)  # (B,256,64,64)
        if self.train_mode == 'finetune-decoder':
            with torch.no_grad():
                image_embeddings = self.image_encoder.image_encoder(img_down)
                
        if self.train_mode == 'finetune-encoder-decoder':
            image_embeddings = self.image_encoder.image_encoder(img_down)
            

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    '''
    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks
      '''
def build_medsam_lite(medsam_lite_checkpoint_path):
    medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
    )
    # %%
    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )
    
    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )
    
    # %%
    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )
    medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu')
    medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
    return medsam_lite_model

if __name__ == '__main__':
    # sam_model = sam_model_registry['vit_b'](checkpoint='sam_vit_b_01ec64.pth')
    # model = MedSAM(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder)
    sam_model = build_medsam_lite('lite_medsam.pth')
    model = MedSAM_Lite_6ch(sam_model.image_encoder, sam_model.mask_decoder, sam_model.prompt_encoder)

    img = torch.rand(2,6,256,256)
    mask = (img[:,:1,:,:]>0.5)*1

    # img = TF.resize(img, (1024, 1024), antialias=True)

    B,_, H, W = mask.shape
    # set the bbox as the image size for fully automatic segmentation
    boxes = torch.from_numpy(np.array([[0,0,W,H]]*B)).float()
                

    mask = model(img, boxes)
    print(mask.shape)

    print(model.image_encoder)
    img_down = model.image_encoder.channel_downsample(img)  # (B,256,64,64)
    image_embeddings = model.image_encoder.image_encoder(img_down)
    print(image_embeddings.shape)



    
        
