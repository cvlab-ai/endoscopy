import os
from typing import List, Optional
from PIL import Image, ImageColor
from src.copy_strategy import AbstractCopyStrategy
from src.structs import MaskRepresentantion, MaskColor

class ImageWriter:

    def __init__(self, img_mode: str, mask_mode: str, copy_strategy: AbstractCopyStrategy) -> None:
        self.img_mode = img_mode
        self.mask_mode = mask_mode
        self.default_copy_strategy = copy_strategy

    def write_frame(self, src: str, dest: str) -> None:
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        img = Image.open(src)
        if (self.img_mode is None or img.mode == self.img_mode):
            self.default_copy_strategy.copy(src, dest)
        else:
            img = img.convert(self.img_mode)
            img.save(dest)

    def write_mask(self, mask_reps: List[MaskRepresentantion], dest: str, base_img_src: str) -> None:
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if len(mask_reps) > 1:
            self.__write_merged_masks(mask_reps, dest, base_img_src)
        elif len(mask_reps) == 1:
            self.__write_single_mask_repr(mask_reps[0], dest, base_img_src)
        else:
            raise "Invalid State: write_masks method called with empty source list."

    def __write_single_mask_repr(self, mask_repr: MaskRepresentantion, dest: str, base_img_src: str) -> None:
        if mask_repr.is_of_color():
            img = self.__create_mask_based_on_frame(color_str=self.__convert_to_pil_color_str(mask_repr.color), desired_mode=self.mask_mode, base_img_src=base_img_src)
            img.save(dest)
        else:
            self.__write_mask_from_path(src=mask_repr.mask_path, dest=dest, base_img_src=base_img_src)


    def __write_mask_from_path(self, src: str, dest: str, base_img_src: str) -> None:
        if os.path.getsize(src) == 0:
            img = self.__create_mask_based_on_frame(color_str='white', desired_mode=self.mask_mode, base_img_src=base_img_src)
            img.save(dest)
        else:
            img = Image.open(src)
            is_img_in_desired_mode = self.mask_mode is None or img.mode == self.mask_mode
            if is_img_in_desired_mode:
                self.default_copy_strategy.copy(src, dest)
            else:
                img = img.convert(self.mask_mode)
                img.save(dest)
            
    def __write_merged_masks(self, mask_reps: List[MaskRepresentantion], dest: str, base_img_src: str) -> None:
        base_img = Image.open(base_img_src)
        desired_size = base_img.size
        desired_mode = self.mask_mode if self.mask_mode is not None else base_img.mode

        img = Image.new(mode='L', size=desired_size, color='black')     
        for mask_repr in mask_reps:
            img_to_merge = self.__prepare_mask_image_to_merge(mask_repr, base_img_src=base_img_src)
            img.paste(im='white', mask=img_to_merge)

        img = img.convert(desired_mode)
        img.save(dest)

    def __prepare_mask_image_to_merge(self, mask_repr: MaskRepresentantion, base_img_src: str) -> Image:
        if mask_repr.is_of_color():
            return self.__create_mask_based_on_frame(color_str=self.__convert_to_pil_color_str(mask_repr.color), desired_mode='L', base_img_src=base_img_src)

        if os.path.getsize(mask_repr.mask_path) == 0:
            return self.__create_mask_based_on_frame(color_str='white', desired_mode='L', base_img_src=base_img_src)
        
        img = Image.open(mask_repr.mask_path)
        img = img.convert('L')
        return img 

    def __create_mask_based_on_frame(self, color_str: str, desired_mode: Optional[str], base_img_src: str) -> Image:
        base_img = Image.open(base_img_src)
        desired_size = base_img.size
        desired_mode = desired_mode if desired_mode is not None else base_img.mode
        color = ImageColor.getcolor(color_str, desired_mode)

        return Image.new(mode=desired_mode, size=desired_size, color=color)

    def __convert_to_pil_color_str(self, mask_color: MaskColor) -> str:
        if mask_color == MaskColor.BLACK:
            return "black"
        if mask_color == MaskColor.WHITE:
            return "white"
        raise f"Invalid State: {mask_color} not supported!"