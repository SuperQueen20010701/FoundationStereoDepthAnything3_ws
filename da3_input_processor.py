from operator import truediv
from PIL import Image
from depth_anything_3.services.input_handlers import (
    ImagesHandler,
    InputHandler,
    parse_export_feat,
)
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.api import DepthAnything3
from typing import List, Tuple
import glob
import os
import cv2
import numpy as np
import time
import math
import torch
import torchvision.transforms as T

def images_input_check(images_dir: str, 
                        image_extensions: str = "png,jpg,jpeg") -> Tuple[bool,List[str]]:
    InputHandler.validate_path(images_dir, "Images directory")
    extensions = [ext.strip().lower() for ext in image_extensions.split(",")]
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
    image_files = []
    for ext in extensions:
        pattern = f"*{ext}"
        # Search recursively in subdirectories using ** pattern
        left_imgs= os.path.join(images_dir,'left','rgb',pattern)
        if InputHandler.validate_path(left_imgs, "Image file"):   
            image_files.append(left_imgs)
        right_imgs= os.path.join(images_dir,'right','rgb',pattern)
        if InputHandler.validate_path(right_imgs, "Image file"):   
            image_files.append(right_imgs)
        if len(image_files) != 2:
            return False,[]
    return True,image_files

def export_layer_parser(export_feat: str) -> List[int]:
    export_feat_layers = parse_export_feat(export_feat)
    return export_feat_layers

#resize image size 

class InputProcessorV2(InputProcessor):
    def __init__(self, divider: int = 16, patch_size: int | None = None) -> None:
        super(InputProcessorV2,self).__init__()
        # We need H,W divisible by BOTH (divider) and (patch_size). Use LCM to unify constraints.
        if patch_size is None:
            patch_size = int(getattr(self, "PATCH_SIZE", 14))
        if divider <= 0 or patch_size <= 0:
            raise ValueError(f"divider and patch_size must be positive, got divider={divider}, patch_size={patch_size}")
        self.divider = int(divider)
        self.PATCH_SIZE = int(math.lcm(self.divider, int(patch_size)))

    def _resize_longest_side(self,img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        if target_size < self.PATCH_SIZE:
            raise ValueError(
                f"process_res(target_size)={target_size} is smaller than required lcm(PATCH_SIZE)={self.PATCH_SIZE}. "
                f"Cannot satisfy 'upper bound' + divisibility simultaneously."
            )
        if longest == target_size:
            # Still ensure divisibility by LCM so downstream is a no-op
            scale = 1.0
        else:
            scale = target_size / float(longest)

        # Keep aspect ratio, and FLOOR to ensure we never exceed the upper bound after making divisible.
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        final_w = max(self.PATCH_SIZE, (new_w // self.PATCH_SIZE) * self.PATCH_SIZE)
        final_h = max(self.PATCH_SIZE, (new_h // self.PATCH_SIZE) * self.PATCH_SIZE)
        if (w, h) == (final_w, final_h):
            return img
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (final_w, final_h), interpolation=interpolation)
        return Image.fromarray(arr)

    def _resize_shortest_side(self,img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        shortest = min(w, h)
        if target_size < self.PATCH_SIZE:
            raise ValueError(
                f"process_res(target_size)={target_size} is smaller than required lcm(PATCH_SIZE)={self.PATCH_SIZE}. "
                f"Cannot satisfy 'lower bound' + divisibility simultaneously."
            )
        if shortest == target_size:
            scale = 1.0
        else:
            scale = target_size / float(shortest)

        # lower_bound_* means we may scale up; still floor-to-multiple to keep divisibility stable.
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        final_w = max(self.PATCH_SIZE, (new_w // self.PATCH_SIZE) * self.PATCH_SIZE)
        final_h = max(self.PATCH_SIZE, (new_h // self.PATCH_SIZE) * self.PATCH_SIZE)
        if (w, h) == (final_w, final_h):
            return img        
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (final_w, final_h), interpolation=interpolation)
        return Image.fromarray(arr)


def _process_inputs(image_files: list[np.ndarray | Image.Image | str],
                    extrinsics: np.ndarray | None = None,
                    intrinsics: np.ndarray | None = None,
                    process_res: int = 504,
                    process_res_method: str = "upper_bound_resize",
                    patch_size : int = 14,
                    *,
                    num_workers: int = 8,
                    print_progress: bool = False,
                    sequential: bool | None = None,
                    desc: str | None = "Preprocess",)-> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    # 处理多张图片
    # resize到能够被14和16整除同时不大于up bounder size的图片
    start_time = time.time()
    input_processor_v2 = InputProcessorV2(divider=16, patch_size=patch_size)
    imgs_cpu, extrinsics, intrinsics = input_processor_v2(
        image_files,
        extrinsics,
        intrinsics,
        process_res,
        process_res_method,
        num_workers=num_workers,
        print_progress=print_progress,
        sequential=sequential,
        desc=desc,
    )
    end_time = time.time()
    logger.info(
        "Processed Images Done taking",
        end_time - start_time,
        "seconds.Image Shape: ",
        imgs_cpu.shape,
    )
    return imgs_cpu, extrinsics, intrinsics

def prepare_model_inputs(imgs_cpu: torch.Tensor,
                        extrinsics: torch.Tensor | None, 
                        intrinsics: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    da3_model = DepthAnything3(model_name="da3mono-large")
    imgs, ex_t, in_t = da3_model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
    ex_t_norm = da3_model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
    return imgs, ex_t_norm, in_t
    
