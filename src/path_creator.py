import os

class PathCreator:
    def __init__(self, root: str, ignore_dataset_type: bool, ignore_dataset_name: bool, ignore_img_type: bool, ignore_class_name: bool):
        self.root = root
        self.ignore_dataset_type = ignore_dataset_type
        self.ignore_dataset_name = ignore_dataset_name
        self.ignore_img_type = ignore_img_type
        self.ignore_class_name = ignore_class_name

    def create(self, dataset_type: str, dataset_name: str, img_type: str, class_name: str, file_name: str):
        return os.path.join(
            self.root, 
            "" if self.ignore_dataset_type else dataset_type,
            "" if self.ignore_dataset_name else dataset_name,
            "" if self.ignore_img_type else img_type,
            "" if self.ignore_class_name else class_name,
            file_name
        )
