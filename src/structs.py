from enum import Enum
from typing import List

class MaskColor(Enum):
    BLACK = "black",
    WHITE = "white"

class MaskRepresentation:
    def __init__(self, mask_path: str, color: MaskColor) -> None:
        self.mask_path = mask_path
        self.color = color

    @staticmethod
    def of_color(color: MaskColor):
        return MaskRepresentation(color=color, mask_path=None)
        
    @staticmethod
    def of_path(mask_path: str):
        return MaskRepresentation(mask_path=mask_path, color=None)

    def is_of_color(self): 
        return self.color is not None

class UnmergedMaskData:
    def __init__(self, class_name: str, path: str) -> None:
        self.class_name = class_name
        self.mask_path = path

class MergedMaskData:
    def __init__(self, class_name: str, repr: List[MaskRepresentation]) -> None:
        self.class_name = class_name
        self.repr = repr