from __future__ import annotations

from typing import Sequence
import cv2
from PIL import Image
import numpy as np
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.logger import logger
import torch
import torchvision.transforms as T
from depth_anything_3.utils.parallel_utils import parallel_execution

class InputProcessorV2(InputProcessor):
    def init(self,divider:int = 16):
        super().__init__()
        self.divider = divider
    def _resize_longest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        scale = target_size / float(longest)
        new_h = w * scale
        new_w = h * scale
        final_w = int(max(1,round(new_w / self.divider) * self.divider))
        final_h = int(max(1, round(new_h / self.divider) * self.divider))
        if (w, h) == (final_w, final_h):
            return img
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (final_w, final_h), interpolation=interpolation)
        return Image.fromarray(arr)

    def _resize_shortest_side(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        shortest = min(w, h)

        scale = target_size / float(shortest)
        new_w = w * scale
        new_h = h * scale
        final_w = int(max(1,round(new_w / self.divider) * self.divider))
        final_h = int(max(1,round(new_h / self.divider) * self.divider))
        if (w, h) == (final_w, final_h):
            return img        
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (final_w, final_h), interpolation=interpolation)
        return Image.fromarray(arr)

    def _resize_image(self, img: Image.Image, target_size: int, method: str) -> Image.Image:
        if method in ("upper_bound_resize", "upper_bound_crop"):
            return self._resize_longest_side(img, target_size)
        elif method in ("lower_bound_resize", "lower_bound_crop"):
            return self._resize_shortest_side(img, target_size)
        else:
            raise ValueError(f"Unsupported resize method: {method}")

    def _process_one(
        self,
        img: np.ndarray | Image.Image | str,
        extrinsic: np.ndarray | None = None,
        intrinsic: np.ndarray | None = None,
        *,
        process_res: int,
        process_res_method: str,
    ) -> tuple[torch.Tensor, tuple[int, int], np.ndarray | None, np.ndarray | None]:
        # Load & remember original size
        logger.info("=" * 80)
        logger.info(f"[InputProcessor] [process_one] img")
        
        pil_img = self._load_image(img)
        orig_w, orig_h = pil_img.size
        logger.info(f"[InputProcessor] [process_one] orig_w: {orig_w}, orig_h: {orig_h} , and after resize image type is {type(pil_img)}")

        # Boundary resize
        pil_img = self._resize_image(pil_img, process_res, process_res_method)
        w, h = pil_img.size
        logger.info(f"After resize the image size is w:{w} and h{h}")
        intrinsic = self._resize_ixt(intrinsic, orig_w, orig_h, w, h)

        # Enforce divisibility by PATCH_SIZE
        if process_res_method.endswith("resize"):
            pil_img = self._make_divisible_by_resize(pil_img, self.PATCH_SIZE)
            new_w, new_h = pil_img.size
            intrinsic = self._resize_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        elif process_res_method.endswith("crop"):
            pil_img = self._make_divisible_by_crop(pil_img, self.PATCH_SIZE)
            new_w, new_h = pil_img.size
            intrinsic = self._crop_ixt(intrinsic, w, h, new_w, new_h)
            w, h = new_w, new_h
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        # Convert to tensor & normalize
        img_tensor = self._normalize_image(pil_img)
        _, H, W = img_tensor.shape
        assert (W, H) == (w, h), "Tensor size mismatch with PIL image size after processing."
        logger.info(f"After process_res_method the new image tensor size is H:{H} W:{W}")
        # Return: (img_tensor, (H, W), intrinsic, extrinsic)
        return img_tensor, (H, W), intrinsic, extrinsic

    def _run_parallel(
        self,
        *,
        image: list[np.ndarray | Image.Image | str],
        exts_list: list[np.ndarray | None] | None,
        ixts_list: list[np.ndarray | None] | None,
        process_res: int,
        process_res_method: str,
        num_workers: int,
        print_progress: bool,
        sequential: bool,
        desc: str | None,
    ):
        results = parallel_execution(
            image,
            exts_list,
            ixts_list,
            action=self._process_one,  # (img, extrinsic, intrinsic, ...)
            num_processes=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
            process_res=process_res,
            process_res_method=process_res_method,
        )
        if not results:
            raise RuntimeError(
                "No preprocessing results returned. Check inputs and parallel_execution."
            )
        return results

    def __call__(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        *,
        num_workers: int = 8,
        print_progress: bool = False,
        sequential: bool | None = None,
        desc: str | None = "Preprocess",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns:
            (tensor, extrinsics_list, intrinsics_list)
            tensor shape: (1, N, 3, H, W)
        """
        sequential = self._resolve_sequential(sequential, num_workers)
        exts_list, ixts_list = self._validate_and_pack_meta(image, extrinsics, intrinsics)

        results = self._run_parallel(
            image=image,
            exts_list=exts_list,
            ixts_list=ixts_list,
            process_res=process_res,
            process_res_method=process_res_method,
            num_workers=num_workers,
            print_progress=print_progress,
            sequential=sequential,
            desc=desc,
        )

        proc_imgs, out_sizes, out_ixts, out_exts = self._unpack_results(results)
        proc_imgs, out_sizes, out_ixts = self._unify_batch_shapes(proc_imgs, out_sizes, out_ixts)

        batch_tensor = self._stack_batch(proc_imgs)
        out_exts = (
            torch.from_numpy(np.asarray(out_exts)).float()
            if out_exts is not None and out_exts[0] is not None
            else None
        )
        out_ixts = (
            torch.from_numpy(np.asarray(out_ixts)).float()
            if out_ixts is not None and out_ixts[0] is not None
            else None
        )
        return (batch_tensor, out_exts, out_ixts)    