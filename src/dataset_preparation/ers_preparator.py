import pandas as pd
import os
from class_mappers import AbstractClassMapper
from typing import List

def generate_dataframe_for_ers(dataset_path: str, class_mapper: AbstractClassMapper, use_seq=True, use_empty_masks=False) -> pd.DataFrame:
    data = []
    for patient_dir in __list_dirs(dataset_path):
        patient_id = os.path.basename(patient_dir)
        for data_dir in __get_data_dirs(patient_dir, use_seq):
            frames_dir = os.path.join(data_dir, "frames")
            labels_dir = os.path.join(data_dir, "labels")
            if not os.path.isdir(labels_dir):
                continue

            for mask_path in __list_files(labels_dir):

                if (not use_empty_masks) and os.path.getsize(mask_path) == 0:
                    continue

                mask_basename = os.path.basename(mask_path)
                img_path = os.path.join(frames_dir, mask_basename[:6] + ".png")
                for class_name in __extract_classes_from_mask_name(mask_basename, class_mapper):
                    data.append(
                        {
                            'patient_id': patient_id,
                            'class': class_name,
                            'img_path': img_path,
                            'mask_path': mask_path,
                            'dataset': 'ers'
                        }
                    )

    return pd.DataFrame(data)

def __extract_classes_from_mask_name(mask_basename: str, class_mapper: AbstractClassMapper) -> List[str]:
    mask_name_without_extension = os.path.splitext(mask_basename)[0]
    unprocessed_class_names = [class_mapper.map(class_name) for class_name in mask_name_without_extension.split('_') if len(class_name) == 3]
    return {class_name for class_name in unprocessed_class_names if class_name is not None}

def __get_data_dirs(patient_dir: str, use_seq: bool) -> List[str]:
    if use_seq:
        return __list_dirs(patient_dir)
    return [os.path.join(patient_dir, "samples")]

def __list_dirs(path: str) -> List[str]:
    return [os.path.join(path, dirname) for dirname in os.listdir(path) if os.path.isdir(os.path.join(path, dirname))]

def __list_files(path: str) -> List[str]:
    return [os.path.join(path, dirname) for dirname in os.listdir(path) if os.path.isfile(os.path.join(path, dirname))]
