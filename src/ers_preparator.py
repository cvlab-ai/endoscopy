import pandas as pd
import os
import yaml
from typing import List, Dict
from src.class_mappers import AbstractClassMapper, DummyClassMapper, DictClassMapper

class ErsPreparator:

    def __init__(self, args) -> None:
        self.dataset_path = args.ers_path
        self.class_mapper = ErsPreparator.__create_class_mapper(args.ers_class_mapper_path)
        self.use_seq = args.ers_use_seq
        self.use_empty_masks = args.ers_use_empty_masks
        self.binary = args.binary
        
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

                frame_to_masks = self.__create_frame_to_masks_dict(frames_dir, labels_dir)
                frame_to_class_records = self.__create_frame_to_class_records_dict(frame_to_masks)
                for frame_path, class_records in frame_to_class_records.items():
                    self.__merge_class_records_and_append_to_data(data, frame_path, class_records, patient_id)

        return pd.DataFrame(data)

    def __create_frame_to_masks_dict(self, frames_dir: str, labels_dir: str) -> Dict:
        frame_to_masks = {}
        frames_paths = self.__list_files(frames_dir)
        masks_paths = self.__list_files(labels_dir)
        for frame_path in frames_paths:
            frame_name = os.path.splitext(os.path.basename(frame_path))[0]
            masks_for_frame = [mask_path for mask_path in masks_paths if os.path.basename(mask_path).startswith(frame_name)]
            frame_to_masks[frame_path] = masks_for_frame
        return frame_to_masks

    def __create_frame_to_class_records_dict(self, frame_to_masks: Dict) -> Dict:
        frame_to_class_records = {}
        for frame_path, masks_paths in frame_to_masks.items():
            class_records_for_frame = []
            for mask_path in masks_paths:
                mask_basename = os.path.basename(mask_path)
                accept_healthy_only = (not self.use_empty_masks) and os.path.getsize(mask_path) == 0
                for class_name in self.__extract_classes_from_mask_name(mask_basename, accept_healthy_only):
                    class_records_for_frame.append({
                        'class': class_name,
                        'mask_path': mask_path,
                    })
            frame_to_class_records[frame_path] = class_records_for_frame
        return frame_to_class_records

    def __merge_class_records_and_append_to_data(self, data: Dict, frame_path: str, class_records: List[Dict], patient_id: str) -> None:
        processed_class_names = set()
        all_classes = {class_record['class'] for class_record in class_records}
        for class_name in all_classes:
            if class_name in processed_class_names:
                continue
            processed_class_names.add(class_name)

            records = [record for record in class_records if record['class'] == class_name]
            is_class_healthy = self.class_mapper.is_healthy(class_name)
            masks_paths_for_class = list(set([record['mask_path'] for record in records]))

            data.append({
                'patient_id': patient_id,
                'class': class_name,
                'img_path': frame_path,
                'masks_paths': masks_paths_for_class,
                'reverse_mask': not (self.binary or (not is_class_healthy)),
                'dataset': 'ers'
            })

    def __extract_classes_from_mask_name(self, mask_basename: str, only_healthy: bool) -> List[str]:
        mask_name_without_extension = os.path.splitext(mask_basename)[0]
        unprocessed_class_names = [class_name for class_name in mask_name_without_extension.split('_') if len(class_name) == 3]

        mapped_class_names = self.__flatten_and_unique([self.class_mapper.map(class_name) for class_name in unprocessed_class_names])
        mapped_class_names = mapped_class_names if not only_healthy else [class_name for class_name in mapped_class_names if self.class_mapper.is_healthy(class_name)]
        return mapped_class_names

    def __get_data_dirs(self, patient_dir: str, use_seq: bool) -> List[str]:
        if use_seq:
            return self.__list_dirs(patient_dir)
        return [os.path.join(patient_dir, "samples")]

    def __list_dirs(self, path: str) -> List[str]:
        return [os.path.join(path, dirname) for dirname in os.listdir(path) if os.path.isdir(os.path.join(path, dirname))]

    def __list_files(self, path: str) -> List[str]:
        return [os.path.join(path, dirname) for dirname in os.listdir(path) if os.path.isfile(os.path.join(path, dirname))]

    def __flatten_and_unique(self, colection):
        return list({item for sublist in colection for item in sublist})

    @staticmethod
    def __create_class_mapper(class_mapper_path: str) -> AbstractClassMapper: 
        if class_mapper_path is None:
            return DummyClassMapper()
        with open(class_mapper_path, "r") as stream:
            return DictClassMapper(yaml.safe_load(stream))
