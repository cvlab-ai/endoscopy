import pandas as pd
import os
import yaml
from typing import List, Tuple
from src.class_mappers import AbstractClassMapper, DummyClassMapper, DictClassMapper

class ErsPreparator:

    def __init__(self, args) -> None:
        self.dataset_path = args.ers_path
        self.class_mapper = ErsPreparator.__create_class_mapper(args.ers_class_mapper_path)
        self.use_seq = args.ers_use_seq
        self.use_empty_masks = args.ers_use_empty_masks
        self.binary = args.binary
        self.healthy_classes = ['h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07', 'b02']
        

    def generate_dataframe(self) -> pd.DataFrame:
        if not self.dataset_path:
            return pd.DataFrame()

        data = []
        for patient_dir in self.__list_dirs(self.dataset_path):
            patient_id = os.path.basename(patient_dir)
            for data_dir in self.__get_data_dirs(patient_dir, self.use_seq):
                frames_dir = os.path.join(data_dir, "frames")
                labels_dir = os.path.join(data_dir, "labels")
                if not os.path.isdir(labels_dir):
                    continue

                for mask_path in self.__list_files(labels_dir):
                    accept_healthy_only = (not self.use_empty_masks) and os.path.getsize(mask_path) == 0
                    mask_basename = os.path.basename(mask_path)
                    img_path = os.path.join(frames_dir, mask_basename[:6] + ".png")
                    for class_name, isHealthy in self.__extract_classes_from_mask_name(mask_basename, accept_healthy_only):
                        data.append(
                            {
                                'patient_id': patient_id,
                                'class': class_name,
                                'img_path': img_path,
                                'mask_path': mask_path,
                                'reverse_mask': not (self.binary or (not isHealthy)),
                                'dataset': 'ers'
                            }
                        )

        return pd.DataFrame(data)

    def __extract_classes_from_mask_name(self, mask_basename: str, only_healthy: bool) -> List[Tuple[str, bool]]:
        mask_name_without_extension = os.path.splitext(mask_basename)[0]
        unprocessed_class_names = [class_name for class_name in mask_name_without_extension.split('_') if len(class_name) == 3]
        if only_healthy:
            unprocessed_class_names = [class_name for class_name in unprocessed_class_names if class_name in self.healthy_classes]
        mapped_class_names = [(self.class_mapper.map(class_name), class_name in self.healthy_classes) for class_name in unprocessed_class_names]
        return {(class_name, isHealthy) for (class_name, isHealthy) in mapped_class_names if class_name is not None}


    def __get_data_dirs(self, patient_dir: str, use_seq: bool) -> List[str]:
        if use_seq:
            return self.__list_dirs(patient_dir)
        return [os.path.join(patient_dir, "samples")]

    def __list_dirs(self, path: str) -> List[str]:
        return [os.path.join(path, dirname) for dirname in os.listdir(path) if os.path.isdir(os.path.join(path, dirname))]

    def __list_files(self, path: str) -> List[str]:
        return [os.path.join(path, dirname) for dirname in os.listdir(path) if os.path.isfile(os.path.join(path, dirname))]

    @staticmethod
    def __create_class_mapper(class_mapper_path: str) -> AbstractClassMapper: 
        if class_mapper_path is None:
            return DummyClassMapper()
        with open(class_mapper_path, "r") as stream:
            return DictClassMapper(yaml.safe_load(stream))
