import pandas as pd
import os
import yaml
from typing import List, Dict
from src.mask_data_merger import MaskDataMerger
from src.structs import UnmergedMaskData
from src.class_mappers import AbstractClassMapper, DummyClassMapper, DictClassMapper

class ErsPreparator:

    def __init__(self, args) -> None:
        self.dataset_path = args.ers_path
        self.class_mapper = ErsPreparator.__create_class_mapper(args.ers_class_mapper_path)
        self.use_seq = args.ers_use_seq
        self.use_empty_masks = args.ers_use_empty_masks
        self.mask_data_merger = MaskDataMerger(args)
        self.acceptable_empty_mask_file_classes = ['h01', 'h02', 'h03', 'h04', 'h05', 'h06', 'h07', 'b02']
        
    def generate_dataframe(self) -> pd.DataFrame:
        if not self.dataset_path:
            return pd.DataFrame()

        data = []
        for patient_dir in self.__list_dirs(self.dataset_path):
            patient_id = os.path.basename(patient_dir)
            for data_dir in self.__get_data_dirs(patient_dir, self.use_seq):
                data_dir_basename = os.path.basename(data_dir)

                frames_dir = os.path.join(data_dir, "frames")
                labels_dir = os.path.join(data_dir, "labels")
                if not os.path.isdir(labels_dir):
                    continue

                frames_paths = self.__list_files(frames_dir)
                masks_paths = self.__list_files(labels_dir)

                for frame_path in frames_paths:
                    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
                    masks_for_frame = [mask_path for mask_path in masks_paths if os.path.basename(mask_path).startswith(frame_name)]
                    unmapped_masks_data = self.__create_masks_data(masks_for_frame)

                    merged_mask_data = self.mask_data_merger.merge(masks_data=unmapped_masks_data, mapper=self.class_mapper)
                    if merged_mask_data is not None:
                        data.append({
                            'dataset': 'ers',
                            'patient_id': patient_id,
                            'frame_path': frame_path,
                            'proposed_name': f"{patient_id}_{data_dir_basename}_{frame_name}.png",
                            'mask_data': merged_mask_data
                        })

        return pd.DataFrame(data)

    def __create_masks_data(self, masks_paths: List[str]) -> List[UnmergedMaskData]:
        unmapped_mask_data = []
        for mask_path in masks_paths:
            mask_classes = self.__extract_classes_from_mask_name(mask_path)

            for mask_class in mask_classes:
                unmapped_mask_data.append(UnmergedMaskData(mask_class, mask_path))
        return unmapped_mask_data


    def __extract_classes_from_mask_name(self, mask_path: str) -> List[str]:
        mask_basename = os.path.basename(mask_path)
        mask_name_without_extension = os.path.splitext(mask_basename)[0]
        class_names = [class_name for class_name in mask_name_without_extension.split('_') if len(class_name) == 3]
        if self.use_empty_masks or os.path.getsize(mask_path) != 0:
            return class_names
        return [class_name for class_name in class_names if class_name in self.acceptable_empty_mask_file_classes]

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
