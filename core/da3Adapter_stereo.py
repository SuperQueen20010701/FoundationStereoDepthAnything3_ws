"""
FoundationStereo adapter for the DepthAnythong V3 backbone 
Stereo Matching with Depth-Anything-3 backbone
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Tuple, Sequence
import argparse
import torch
import torch.nn as nn
import huggingface_hub,time
import glob
from PIL import Image
import numpy as np


model_dir = '/DATA/disk0/zhaobojun/depthanything_model'
# add the depthanything v3 package into the system pat
def _ensure_da3_on_syspath() -> bool:
    """Ensure `Depth-Anything-3/src` is importable when running from this workspace."""
    try :
        import depth_anything_3
        return True
    except ImportError:
        print(f"WARNNING  !! [Package] depth_anything_3 import error !! using the local library instead")
        file_path = Path(__file__).resolve()
        path = file_path.split('/')
        if 'workspace_ws' in path:
            idx = path.index('workspace_ws')
            workspace_path = '/'.join(path[:idx + 1])
            da3_src_path = workspace_path / 'src' / 'Depth-Anything-3' / 'src'
            sys.path.insert(0, str(da3_src_path))
            return True
        else:
            raise FileNotFoundError(f"workspace_ws not found in {file_path}")
try:
    if  _ensure_da3_on_syspath():
        from depth_anything_3.logger import logger
        from depth_anything_3.services.input_handlers import (
            InputHandler,
            parse_export_feat,
        )
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.model.da3 import DepthAnything3Net
        from .InputProcessor import InputProcessorV2
        # import foundationstereo base model
        from core.extractor import *
except Exception as e:
    print(f"ERROR !! [Package] depth_anything_3 import error !! {e}")
    raise e
# for the parse of the input arguments
def _parse_args() :
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('images_dir',help='the images pair root dir for training')
    parser.add_argument('model_dir',type=str ,default=f'{model_dir}',help='pretrained model path')
    parser.add_argument('--image_extensions',type =str,default='png,jpg,jpeg',help='Comma-separated image file extensions to process')
    parser.add_argument('--work_dir', help='the dir for the dir to save logs and models')
    parser.add_argument('--export_feat',type=str,default="",help="Comma-separated feature layer indices to export (e.g. '0,1,2').")
    # foundationstereo parameter
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')    
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use (default: 0)')
    # depthanything v3 parameter 
    parser.add_argument('--process_res',type=int,default=504,help='Processing resolution')
    parser.add_argument('--process_res_method',type=str,default='upper_bound_resize',help='Processing resolution method')

    args = parser.parse_args()
    return args

def set_device_available(gpu :int =0):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        torch.cuda.empty_cache()
        logger.info(f"Using GPU {gpu}: {torch.cuda.get_device_name(gpu)}")
    else:
        logger.warning("CUDA not available, using CPU instead")
    
    #phase training or inference
    torch.autograd.set_grad_enabled(False)

class FeatureAdapter(nn.Module , huggingface_hub.PyTorchModelHubMixin):
    def __init__(self,model_dir:str,
                gpu:int=0,
                input_layout :str = "image_stacked",
                ref_view_strategy:str = "saddle_balanced",
                process_res:int=504,
                process_res_method:str="upper_bound_resize",):
        super().__init__()
        self.device = gpu
        self.api_da3 = DepthAnything3.from_pretrained(args.model_dir).to(args.gpu)
        self.api_da3.eval()
        self.dinov2_backbone = self.api_da3.model.backbone
        self.dpt_head = self.api_da3.model.head
        self.patch_size = DepthAnything3Net.PATCH_SIZE
        self.ref_view_strategy = ref_view_strategy
        self.input_layout = input_layout
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.down_ratio = int(getattr(self.dpt_head, "down_ratio", 1))
        self.divider = np.lcm(self.patch_size,16)
        self.input_processor = InputProcessorV2(self.divider)
    # foundationstereo input [b,c,h,w]
    # depthanything input [b,s,c,h,w]

    # input depthanything v3 input[h,w]
    def _feature_flat(self, feats: List[torch.Tensor],
                        H:int,W:int) :
        if len(feats) < 4:
            logger.error(f"ERROR !! [FeatureAdapter] [feature_flat] feature export layer is not equal to 4")
        token0 = feats[0][0]
        if token0.ndim != 4:
            logger.error(f"ERROR !! [FeatureAdapter] [feature_flat] token0 shape not equal to 4")
        
        Bpair, S, N, C = token0.shape
        if S != 2:
            logger.error(f"ERROR !! [FeatureAdapter] [feature_flat] image pair is not equal to 2")
        
        # patch size validation 
        ph , pw = H // self.patch_size, W // self.patch_size
        if ph * pw != N:
            logger.error(f"ERROR !! [FeatureAdapter] [feature_flat] token length mismatch: N={N}, ph*pw={ph*pw}")
            raise
        # flatten the feature
        feat_flat = [f[0].reshape(Bpair * S ,N,C) for f in feats[:4]] # flate the feature
        return feat_flat,Bpair,S

    def _resize_feature(self,feat:List[torch.Tensor],H:int ,W:int)-> torch.Tensor:
        def reshape_feature(x,H,W):
            B,_,C = x.shape
            ph ,pw = H // self.patch_size, W // self.patch_size
            x = x.permute(0,2,1).contiguous().reshape(B,C,ph,pw)
            return x
        def project_resize_feature(x,H,W,stage_idx): # 1024 -> 128
            x = self.dpt_head.norm(x)
            x = reshape_feature(x,H,W)
            x = self.dpt_head.projects[stage_idx](x)
            if bool(getattr(self.dpt_head,"pos_embed",False)):
                x = self.dpt_head._add_pos_embed(x,W,H)
            # get refined 256 channel sized feature 
            x = self.dpt_head.resize_layers[stage_idx](x)
            return x
        
        resized_feats = []
        for stage_idx in range(4):
            x = feat[stage_idx]
            x = project_resize_feature(x,H,W,stage_idx)
            resized_feats.append(x)

        fused = self.dpt_head._fuse(resized_feats)
        if isinstance(fused,(tuple, list)):
            fused = fused[0]

        if fused.shape[1] != 128 and hasattr(self.dpt_head, "scratch") and hasattr(self.dpt_head.scratch, "output_conv1"):
            fused = self.head.scratch.output_conv1(fused) # resize channel 256 ->128
        return resized_feats
    
    def _interpoplation_img_size(self,x:torch.Tensor,W:int,H:int)-> torch.Tensor:
        ph,pw = x.shape[-2],x.shape[-1]
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        x = self.dpt_head.custom_interpolate(x, (h_out, w_out), mode="bilinear", align_corners=True)
        if bool(getattr(self.dpt_head,"pos_embed",False)):
            x = self.dpt_head._add_pos_embed(x,W,H)
        return x

    @torch.no_grad()
    def forward(self, image_files: list[np.ndarray | Image.Image | str],
                      intrinsics: np.ndarray | None = None,
                      extrinsics: np.ndarray | None = None,
                      export_feat_layers:Sequence[int] | None = None,**kwargs):

        if image_files!=None:
            img_files = image_files
            for single_img_pair in img_files:
                image_single_pair = [single_img_pair[0],single_img_pair[1]]
                imgs_cpu, extrinsics, intrinsics =  self.input_processor(
                    image_single_pair, extrinsics,  intrinsics, self.process_res, self.process_res_method)
                imgs, ex_t, in_t = self.depthanything3.v(imgs_cpu, extrinsics, intrinsics)
                ex_t_norm = self.depthanything3._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
                export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []
                H,W = imgs.shape[-2],imgs.shape[-1]
                device = imgs.device
                need_sync = device.type == "cuda"
                if need_sync:
                    torch.cuda.synchronize(device)
                start_time = time.time()
                autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                
                with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):
                    if extrinsics is not None:
                        with torch.autocast(device_type=imgs.device.type, enabled=False):
                            cam_token = self.cam_enc(ex_t_norm, in_t, imgs.shape[-2:])
                    else:
                        cam_token = None
                    # output the backbone feature 
                    feats, aux_feats = self.dinov2_backbone(
                                                imgs, 
                                                cam_token=cam_token, 
                                                export_feat_layers=export_feat_layers,
                                                ref_view_strategy=args.ref_view_strategy)

                    # output the depthanthing v3 feature []
                flate_feature,Bpair,S = self._feature_flat(feats,H,W) # flate the feature 
                # resize the channel and image size 
                resized_feature = self._resize_feature(flate_feature,H=H,W=W)
                interpoplation_feature = self._interpoplation_img_size(resized_feature,W=W,H=H)
                fused_feature = interpoplation_feature.view(Bpair,S,interpoplation_feature.shape[1],interpoplation_feature.shape[2],interpoplation_feature.shape[3])
                return fused_feature

def InputHandlerProcess(images_dir:str,image_extensions:str='png,jpg,jpeg',export_feat:str='') ->Tuple[List[List[str,str]],List[int]]:
    InputHandler.validate_path(images_dir, "Images directory")
    extensions = [ext.strip().lower() for ext in image_extensions.split(",")]
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
    image_files = []
    left_imgs :List[str] = []
    right_imgs :List[str] = []
    for ext in extensions:
        pattern = f"*{ext}"
        # Search recursively in subdirectories using ** pattern
        left_imgs.extend(glob.glob(os.path.join(images_dir,'left','rgb',pattern),recursive=True))
        right_imgs.extend(glob.glob(os.path.join(images_dir,'right','rgb',pattern),recursive=True))
        if len(left_imgs) == len(right_imgs):
            image_files = [[l_i,r_i] for l_i,r_i in zip(left_imgs,right_imgs)]
        else:
            raise ValueError(f"the number of left and right images are not equal")
    # process export feature layer
    export_feat_layers = parse_export_feat(export_feat)
    return image_files,export_feat_layers



if __name__ == "__main__":
    # parse arguments
    args = _parse_args()
    # set device
    set_device_available(args.gpu)
    # inference
    image_files : List[List[str,str]] = []
    export_feat_layers : List[int] = []
    image_files, export_feat_layers = InputHandlerProcess(
        args.images_dir, args.image_extensions, args.export_feat
    )


