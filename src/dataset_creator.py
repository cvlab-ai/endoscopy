import os
import shutil
import pandas as pd
from typing import Tuple, Hashable
from src.training_type import TrainingType
from src.path_creator import PathCreator
from src.ers_preparator import ErsPreparator
from src.hyperkvasir_preparator import HyperkvasirPreparator
from src.image_writer import ImageWriter
from src.splitter import DataSplitter

class DatasetCreator:

    def __init__(self, args) -> None:
        self.path_creator = DatasetCreator.__prepare_path_creator(args)
        self.image_writer = DatasetCreator.__prepare_image_writer(args)
        self.data_splitter = DatasetCreator.__prepare_data_splitter(args)

        self.ers_preparator = ErsPreparator(args)
        self.hkvs_preparator = HyperkvasirPreparator(args)

        self.copy_mask = args.training_type == TrainingType.SEGMENTATION,

    def create(self) -> None:
        df = self.__generate_dataframes()
        train_df, val_df, test_df = self.data_splitter.split_and_prepare(df)

        print(f"Generated {train_df.shape[0]} train records, {val_df.shape[0]} validation records and {test_df.shape[0]} test records.")

        self.__fill_output_dir(train_df, 'train')
        self.__fill_output_dir(val_df, 'validation')
        self.__fill_output_dir(test_df, 'test')

        print("Dataset prepared")

    def __fill_output_dir(self, df: pd.DataFrame, type: str):
        for loop_index, data in enumerate(df.iterrows()):
            self.__generate_output_record(data, type)
        
            if loop_index % 100 == 0 and loop_index > 0:
                print(f"Processed {loop_index} images")
            
        print(f"Processed all images from {type} dataset")

    def __generate_output_record(self, data: Tuple[Hashable, pd.Series], type: str) -> None:
        (row_id, record) = data

        dataset_name = record['dataset']
        class_name = record['class']
        src_img_path = record['img_path']
        img_ext = os.path.splitext(src_img_path)[1]
        dest_img_name = f"{row_id}{img_ext}"

        dest_img_path = self.path_creator.create(dataset_type=type, dataset_name=dataset_name, img_type='images', class_name=class_name, file_name=dest_img_name)
        self.image_writer.write_image(src_img_path, dest_img_path)
        
        if self.copy_mask:
            src_mask_path = record['mask_path']
            reverse_color = record['reverse_mask']

            dest_mask_path = self.path_creator.create(dataset_type=type, dataset_name=dataset_name, img_type='masks', class_name=class_name, file_name=dest_img_name)
            self.image_writer.write_mask(src_mask_path, dest_mask_path, reverse_color=reverse_color, base_img_src=src_img_path)

    
    def __generate_dataframes(self) -> pd.DataFrame:
        return pd.concat([
            self.ers_preparator.generate_dataframe(),
            self.hkvs_preparator.generate_dataframe()])

    
    @staticmethod
    def __prepare_path_creator(args) -> PathCreator:
        clean_output = args.force
        output_path = args.output_path
        if clean_output and os.path.isdir(output_path):
            print(f"Cleaning output dir: {output_path}")
            shutil.rmtree(output_path)
            print("Output path cleaned")

        return PathCreator(
            output_path,
            ignore_dataset_type=args.path_ignore_dataset_type,
            ignore_dataset_name=args.path_ignore_dataset_name,
            ignore_img_type=args.path_ignore_img_type,
            ignore_class_name=args.path_ignore_class_name)

    @staticmethod
    def __prepare_image_writer(args) -> ImageWriter:
        img_mode = args.img_mode
        mask_mode = args.mask_mode
        copy_strategy=args.copy_strategy

        return ImageWriter(
            img_mode=img_mode,
            mask_mode=mask_mode,
            copy_strategy=copy_strategy.create())
    
    @staticmethod
    def __prepare_data_splitter(args) -> DataSplitter:
        return DataSplitter(
            train_part=args.train_size,
            val_part=args.validation_size)