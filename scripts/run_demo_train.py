# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from ast import arg
import os,sys
import argparse
import imageio
import torch
import logging
import cv2
import numpy as np
import open3d as o3d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo
from PIL import Image
from core.da3_input_processor import (
    _process_inputs,
    images_input_check,
    export_layer_parser,
    prepare_model_inputs,
)
if __name__=="__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path (supports OpenStereo checkpoint_epoch_*.pth format)')
    parser.add_argument('--cfg_file', type=str, default=None, help='optional config file path (if not provided, will try to find cfg.yaml in checkpoint directory)')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=0, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')

    parser.add_argument("--use_da3", action="store_true", help="Use Depth-Anything-3 as dense feature backbone")
    parser.add_argument("--da3_model_dir", type=str, default="", help="Depth-Anything-3 pretrained model dir (required if --use_da3)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image_extensions", type=str, default="png,jpg,jpeg")
    parser.add_argument("--export_feat", type=str,default="", help="Export features from specified layers using comma-separated indices (e.g., '0,1,2').")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--process_res_method", type=str, default="upper_bound_resize")
    parser.add_argument("--ref_view_strategy", type=str, default="saddle_balanced")
    parser.add_argument("--images_dir", type=str, default="", help="Images directory (required if --use_da3)")
    parser.add_argument("--export_dir", type=str, default="", help="Output directory (required if --use_da3)")
    parser.add_argument("--auto_cleanup", type=str, default="", help="Export directory (required if --use_da3)")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = args.ckpt_dir
    # Try to load config file
    cfg = None
    if args.cfg_file and os.path.exists(args.cfg_file):
        cfg = OmegaConf.load(args.cfg_file)
        logging.info(f"Loaded config from {args.cfg_file}")
    else:
        # Try to find cfg.yaml in checkpoint directory or parent directory
        cfg_paths = [
            f'{os.path.dirname(ckpt_dir)}/cfg.yaml',  # Parent directory
            f'{os.path.dirname(os.path.dirname(ckpt_dir))}/cfg.yaml',  # Grandparent (for OpenStereo output structure)
        ]
        # Also try to find any .yaml file in the checkpoint directory (OpenStereo saves config there)
        ckpt_parent = os.path.dirname(ckpt_dir)
        if os.path.exists(ckpt_parent):
            yaml_files = [f for f in os.listdir(ckpt_parent) if f.endswith('.yaml') or f.endswith('.yml')]
            if yaml_files:
                cfg_paths.append(os.path.join(ckpt_parent, yaml_files[0]))
        
        for cfg_path in cfg_paths:
            if os.path.exists(cfg_path):
                cfg = OmegaConf.load(cfg_path)
                logging.info(f"Loaded config from {cfg_path}")
                break
        
        if cfg is None:
            logging.warning(f"Config file not found. Trying to load from checkpoint or using defaults")
            # Try to load config from checkpoint if available
            try:
                ckpt_temp = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt_temp = torch.load(ckpt_dir, map_location="cpu")
            if 'cfg' in ckpt_temp:
                cfg = ckpt_temp['cfg']
                logging.info("Loaded config from checkpoint")
            else:
                # Use default config structure - you may need to adjust this based on your model
                cfg = OmegaConf.create({})
                logging.warning("Using default config. Some parameters may need to be set manually via command line arguments.")
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)

    try:
        ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch without `weights_only`
        ckpt = torch.load(ckpt_dir, map_location="cpu")
    
    # Log checkpoint info (handle different checkpoint formats)
    if 'global_step' in ckpt:
        logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt.get('epoch', 'N/A')}")
    elif 'epoch' in ckpt:
        logging.info(f"ckpt epoch:{ckpt['epoch']}")
    else:
        logging.info("Loading checkpoint (format may be different)")
    
    # Support multiple checkpoint formats:
    # 1. OpenStereo format: {'model_state': ..., 'epoch': ..., ...}
    # 2. FoundationStereo format: {'model': ..., 'global_step': ..., 'epoch': ..., ...}
    # 3. Direct state_dict
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
        logging.info("Loaded checkpoint in OpenStereo format (model_state key)")
    elif 'model' in ckpt:
        state_dict = ckpt['model']
        logging.info("Loaded checkpoint in FoundationStereo format (model key)")
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values() if isinstance(v, torch.Tensor)):
        # Check if checkpoint is directly a state_dict
        state_dict = ckpt
        logging.info("Loaded checkpoint as direct state_dict")
    else:
        # Fallback: try to use checkpoint as state_dict
        state_dict = ckpt
        logging.warning("Could not identify checkpoint format, attempting to load as state_dict")
    if bool(getattr(args, "use_da3", False)):
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        # These missing keys are expected when swapping the dense backbone to DA3.
        expected_missing_prefixes = (
            "feature.feature_backbone.",
            "feature.feature_backbone_api.",  # just in case of older naming
        )
        missing_other = [k for k in missing if not k.startswith(expected_missing_prefixes)]
        logging.warning(
            "Loaded checkpoint with strict=False because --use_da3 is enabled (architecture differs). "
            f"missing_keys={len(missing)} (non-DA3 expected: {len(missing)-len(missing_other)}), "
            f"unexpected_keys={len(unexpected)}."
        )
        if missing_other:
            logging.warning(
                "Some missing keys are NOT DA3-related. The checkpoint may be incompatible with this code version. "
                f"Example missing keys: {missing_other[:20]}"
            )
    else:
        model.load_state_dict(state_dict, strict=True)

    model.cuda()
    model.float()
    model.eval()

    if args.use_da3 and bool(args.images_dir):
        result, image_files = images_input_check(args.images_dir, args.image_extensions)
        if not result:
            raise ValueError(f"Failed to find matching left and right image pairs in {args.images_dir}")
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {args.images_dir}")
        if len(image_files) % 2 != 0:
            logging.warning(f"Odd number of images found ({len(image_files)}), may cause issues with stereo processing")

        scale = args.scale
        export_feat_layers = export_layer_parser(args.export_feat)
        if export_feat_layers is None:
            print(f"No valid feature layers to export")
        # process image
        imgs_cpu, extrinsics, intrinsics = _process_inputs(image_files, 
                                                            patch_size=model.patch_size,
                                                            process_res=args.process_res, 
                                                            process_res_method=args.process_res_method,)
        imgs,ex_t_norm,in_t = prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
        # Ensure DA3-prepared tensors live on the same device as the model (typically CUDA).
        # `prepare_model_inputs` may return CPU tensors depending on DA3 implementation/config.
        model_device = next(model.parameters()).device
        imgs = imgs.to(model_device, non_blocking=True)
        if ex_t_norm is not None:
            ex_t_norm = ex_t_norm.to(model_device, non_blocking=True)
        if in_t is not None:
            in_t = in_t.to(model_device, non_blocking=True)
        device = imgs.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        with torch.no_grad():
            disp = model.forward_da3_inputs(
                imgs,
                ex_t_norm,
                in_t,
                iters=int(getattr(args, "valid_iters", 32)),
                test_mode=True,
                low_memory=bool(getattr(args, "low_memory", False)),
                hiera=bool(getattr(args, "hiera", 0)),
                small_ratio=0.5,
            )
        # Use processed left image as point cloud color reference (uint8 HxWx3).
        # imgs_cpu is typically uint8 on CPU; fall back robustly.
        try:
            img0_ori = imgs_cpu[0, 0].permute(1, 2, 0).detach().cpu().numpy()
            if img0_ori.dtype != np.uint8:
                img0_ori = np.clip(img0_ori, 0, 255).astype(np.uint8)
        except Exception:
            img0_ori = None
        # Normalize DA3 branch output to numpy (H,W) for post-processing & point cloud
        # disp: [B,1,H,W] (typically B=1)
        disp_t = disp.float().detach().cpu()
        if disp_t.ndim == 4:
            disp_t = disp_t[0, 0]
        elif disp_t.ndim == 3:
            disp_t = disp_t[0]
        disp = disp_t.numpy()
        H, W = disp.shape[-2], disp.shape[-1]
        vis = vis_disparity(disp)
        imageio.imwrite(f'{args.out_dir}/vis.png', vis)
        logging.info(f"Output saved to {args.out_dir}")
    else:
        code_dir = os.path.dirname(os.path.realpath(__file__))
        img0 = imageio.imread(args.left_file)
        img1 = imageio.imread(args.right_file)
        scale = args.scale
        assert scale<=1, "scale must be <=1"
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
        H,W = img0.shape[:2]
        img0_ori = img0.copy()
        logging.info(f"img0: {img0.shape}")

        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H,W)
        vis = vis_disparity(disp)
        vis = np.concatenate([img0_ori, vis], axis=1)
        imageio.imwrite(f'{args.out_dir}/vis.png', vis)
        logging.info(f"Output saved to {args.out_dir}")

    if args.remove_invisible:
        yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = us_right<0
        disp[invalid] = np.inf

    if args.get_pc:
        if img0_ori is None:
            raise ValueError("Point cloud export requires a left RGB image reference, but img0_ori is None.")
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
            baseline = float(lines[1])
            K[:2] *= scale
            depth = K[0,0]*baseline/disp
            np.save(f'{args.out_dir}/depth_meter.npy', depth)
            xyz_map = depth2xyzmap(depth, K)
            pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
            keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
            keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
            pcd = pcd.select_by_index(keep_ids)
            o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
            logging.info(f"PCL saved to {args.out_dir}")

        if args.denoise_cloud:
            logging.info("[Optional step] denoise point cloud...")
            cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
            inlier_cloud = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
            pcd = inlier_cloud

        logging.info("Visualizing point cloud. Press ESC to exit.")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
        vis.run()
        vis.destroy_window()

