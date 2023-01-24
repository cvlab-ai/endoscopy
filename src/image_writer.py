import os
import numpy as np
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
    
    def write_mask(self, src: str, dest: str, reverse_color: bool, base_img_src: str) -> None:
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if os.path.getsize(src) == 0:
            base_img = Image.open(base_img_src)
            desired_size = base_img.size
            desired_mode = self.mask_mode if self.mask_mode is not None else base_img.mode
            color = ImageColor.getcolor('white' if reverse_color else 'black', desired_mode)
            img = Image.new(mode=desired_mode, size=desired_size, color=color)
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


