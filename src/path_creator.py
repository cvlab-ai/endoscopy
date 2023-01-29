import os
from typing import Optional

class SegmentationPathCreator:
    def __init__(self, root: str, ignore_dataset_type: bool, ignore_dataset_name: bool):
        self.root = root
        self.ignore_dataset_type = ignore_dataset_type
        self.ignore_dataset_name = ignore_dataset_name
        

    def create_frame_path(self, dataset_type: str, dataset_name: str, file_name: str):
        return os.path.join(
            self.root, 
            "" if self.ignore_dataset_type else dataset_type,
            "" if self.ignore_dataset_name else dataset_name,
            "images",
            file_name
        )

    def create_mask_path(self, dataset_type: str, dataset_name: str, class_name: Optional[str], file_name: str):
        return os.path.join(
            self.root, 
            "" if self.ignore_dataset_type else dataset_type,
            "" if self.ignore_dataset_name else dataset_name,
            "masks",
            class_name if class_name is not None else "",
            file_name
        )


class ClassificationPathCreator:
    def __init__(self, root: str, ignore_dataset_type: bool, ignore_dataset_name: bool):
        self.root = root
        self.ignore_dataset_type = ignore_dataset_type
        self.ignore_dataset_name = ignore_dataset_name
        

    def create_frame_path(self, dataset_type: str, dataset_name: str, class_name: str, file_name: str):
        return os.path.join(
            self.root, 
            "" if self.ignore_dataset_type else dataset_type,
            "" if self.ignore_dataset_name else dataset_name,
            class_name,
            file_name
        )