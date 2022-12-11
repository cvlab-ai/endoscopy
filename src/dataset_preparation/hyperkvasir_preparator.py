import pandas as pd
import os

from typing import List
from enums import TrainingType


def generate_dataframe_for_hyperkvasir(dataset_path: str, training_type: TrainingType) -> pd.DataFrame:
    if training_type == TrainingType.CLASSIFICATION:
        return __hyperkvasir_classification(dataset_path)
    if training_type == TrainingType.SEGMENTATION:
        return __hyperkvasir_segmentation(dataset_path)


def __hyperkvasir_classification(dataset_path: str) -> pd.DataFrame:
    data = []
    labeled_images_path = os.path.join(dataset_path, "labeled-images")
    for gi_tract in __list_dirs(labeled_images_path):
        gi_track_path = os.path.join(labeled_images_path, gi_tract)
        for finding_type in __list_dirs(gi_track_path):
            finding_type_path = os.path.join(gi_track_path, finding_type)
            for pathology in __list_dirs(finding_type_path):
                pathology_path = os.path.join(finding_type_path, pathology)
                for sample in __list_files(pathology_path):
                    data.append(
                        {
                            'patient_id': None,
                            'class': pathology,
                            'img_path': os.path.join(pathology_path, sample),
                            'mask_path': None,
                            'dataset': "hyperkvasir"
                        }
                    )

    return pd.DataFrame(data)


def __hyperkvasir_segmentation(dataset_path: str) -> pd.DataFrame:
    segmented_images_path = os.path.join(dataset_path, "segmented-images")
    masks_path = os.path.join(segmented_images_path, "masks")
    images_path = os.path.join(segmented_images_path, "images")

    images = __list_files(images_path)
    masks = __list_files(masks_path)
    filenames = list(set(masks).intersection(images))

    data = []
    for filename in filenames:
        data.append(
            {
                'patient_id': None,
                'class': "Polyps",
                'img_path': os.path.join(images_path, filename),
                'mask_path': os.path.join(masks_path, filename),
                'dataset': "hyperkvasir"
            }
        )

    return pd.DataFrame(data)


def __list_dirs(path: str) -> List[str]:
    return [dirname for dirname in os.listdir(path) if os.path.isdir(os.path.join(path, dirname))]


def __list_files(path: str) -> List[str]:
    return [filename for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename))]
    