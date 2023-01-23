import pandas as pd
import os
from typing import List


class HyperkvasirPreparator:

    def __init__(self, args) -> None:
        self.binary = args.binary
        self.dataset_path = args.hyperkvasir_path

    def generate_dataframe(self) -> pd.DataFrame:
        if not self.dataset_path:
            return pd.DataFrame()

        segmented_images_path = os.path.join(self.dataset_path, "segmented-images")
        masks_path = os.path.join(segmented_images_path, "masks")
        images_path = os.path.join(segmented_images_path, "images")

        images = self.__list_files(images_path)
        masks = self.__list_files(masks_path)
        filenames = list(set(masks).intersection(images))

        data = []
        for filename in filenames:
            data.append(
                {
                    'patient_id': None,
                    'class': "polyp",
                    'img_path': os.path.join(images_path, filename),
                    'mask_path': os.path.join(masks_path, filename),
                    'reverse_mask': not self.binary,
                    'dataset': "hyperkvasir"
                }
            )

        return pd.DataFrame(data)

    def __list_files(self, path: str) -> List[str]:
        return [filename for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename))]
        