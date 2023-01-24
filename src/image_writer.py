import os
import numpy as np
from typing import List
from PIL import Image, ImageColor
from src.copy_strategy import AbstractCopyStrategy

class ImageWriter:

    def __init__(self, img_mode: str, mask_mode: str, copy_strategy: AbstractCopyStrategy) -> None:
        self.img_mode = img_mode
        self.mask_mode = mask_mode
        self.default_copy_strategy = copy_strategy

    def write_image(self, src: str, dest: str) -> None:
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        img = Image.open(src)
        if (self.img_mode is None or img.mode == self.img_mode):
            self.default_copy_strategy.copy(src, dest)
        else:
            img = img.convert(self.img_mode)
            img.save(dest)
    
    def write_masks(self, src: List[str], dest: str, reverse_color: bool, base_img_src: str) -> None:
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if len(src) > 1:
            self.__write_merged_masks(src, dest, reverse_color=reverse_color, base_img_src=base_img_src)
        elif len(src) == 1:
            self.__write_mask(src[0], dest, reverse_color=reverse_color, base_img_src=base_img_src)
        else:
            raise "Invalid State: write_masks method called with empty source list."

    def __write_mask(self, src: str, dest: str, reverse_color: bool, base_img_src: str) -> None:
        if os.path.getsize(src) == 0:
            color_str = 'white' if reverse_color else 'black'
            img = self.__create_mask_based_on_img(base_img_src=base_img_src, desired_mode=self.mask_mode, color_str=color_str)
            img.save(dest)
        else:
            img = Image.open(src)
            is_img_in_desired_mode = self.mask_mode is None or img.mode == self.mask_mode
            if is_img_in_desired_mode and not reverse_color:
                self.default_copy_strategy.copy(src, dest)
            else:
                output_mode = self.mask_mode if self.mask_mode is not None else img.mode
                img = self.reverse_mask(img) if reverse_color else img
                img = img.convert(output_mode)
                img.save(dest)

    def __write_merged_masks(self, src: List[str], dest: str, reverse_color: bool, base_img_src: str) -> None:
        base_img = Image.open(base_img_src)
        desired_size = base_img.size
        desired_mode = self.mask_mode if self.mask_mode is not None else base_img.mode

        img = Image.new(mode='L', size=desired_size, color='black')     

        for src_mask in src:
            img_to_merge = self.__prepare_mask_image_to_merge(src_mask, reverse_color=reverse_color, base_img_src=base_img_src)
            img.paste(im='white', mask=img_to_merge)

        img = img.convert(desired_mode)
        img.save(dest)

    def __prepare_mask_image_to_merge(self, src: str, reverse_color: bool, base_img_src: str) -> Image:
        if os.path.getsize(src) == 0:
            color_str = 'white' if reverse_color else 'black'
            return self.__create_mask_based_on_img(base_img_src=base_img_src, desired_mode='L', color_str=color_str)
        
        img = Image.open(src)
        img = self.reverse_mask(img) if reverse_color else img
        img = img.convert('L')
        return img

    def __create_mask_based_on_img(self, base_img_src: str, desired_mode: str, color_str: str) -> Image:
        base_img = Image.open(base_img_src)
        desired_size = base_img.size
        desired_mode = desired_mode if desired_mode is not None else base_img.mode
        color = ImageColor.getcolor(color_str, desired_mode)
        return Image.new(mode=desired_mode, size=desired_size, color=color)        

    #Copied from: https://stackoverflow.com/a/3753428
    def reverse_mask(self, img: Image) -> Image:
        img = img.convert('RGBA')

        data = np.array(img)   # "data" is a height x width x 4 numpy array
        red, green, blue, _ = data.T

        white_areas = (red == 255) & (blue == 255) & (green == 255)
        black_areas = (red == 0) & (blue == 0) & (green == 0)
        data[..., :-1][white_areas.T] = (0, 0, 0)
        data[..., :-1][black_areas.T] = (255, 255, 255)

        return Image.fromarray(data)
