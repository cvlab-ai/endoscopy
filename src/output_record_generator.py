import os
import shutil
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Hashable
from src.training_type import TrainingType
from src.image_writer import ImageWriter
from src.path_creator import SegmentationPathCreator, ClassificationPathCreator


class OutputRecordGenerator(ABC):
    @abstractmethod
    def generate_output_record(self, data: Tuple[Hashable, pd.Series], type: str) -> None:
        raise NotImplementedError

    @staticmethod
    def clean_output_dir(output_path: str) -> None:
        if os.path.isdir(output_path):
            print(f"Cleaning output dir: {output_path}")
            shutil.rmtree(output_path)
            print("Output path cleaned")

    @staticmethod
    def prepare_image_writer(args) -> ImageWriter:
        img_mode = args.img_mode
        mask_mode = args.mask_mode
        copy_strategy=args.copy_strategy

        return ImageWriter(
            img_mode=img_mode,
            mask_mode=mask_mode,
            copy_strategy=copy_strategy.create())


class SegmentationOutputRecordGenerator(OutputRecordGenerator):
    def __init__(self, args) -> None:
        self.binary = args.training_type == TrainingType.BINARY_SEG
        self.path_creator = SegmentationOutputRecordGenerator.__prepare_path_creator(args)
        self.image_writer = OutputRecordGenerator.prepare_image_writer(args)

    def generate_output_record(self, data: Tuple[Hashable, pd.Series], type: str) -> None:
        (_, record) = data
        dataset_name = record['dataset']
        frame_path = record['frame_path']
        dest_frame_name = record['proposed_name']
        masks_data = record['mask_data']

        dest_frame_path = self.path_creator.create_frame_path(dataset_type=type, dataset_name=dataset_name, file_name=dest_frame_name)
        self.image_writer.write_frame(frame_path, dest_frame_path)

        for mask_data in masks_data:
            class_name = mask_data.class_name if not self.binary else None
            dest_mask_path = self.path_creator.create_mask_path(dataset_type=type, dataset_name=dataset_name, class_name=class_name, file_name=dest_frame_name)
            self.image_writer.write_mask(mask_data.repr, dest_mask_path, base_img_src=frame_path)

    @staticmethod
    def __prepare_path_creator(args) -> SegmentationPathCreator:
        clean_output = args.force
        output_path = args.output_path
        if clean_output:
            OutputRecordGenerator.clean_output_dir(output_path)

        return SegmentationPathCreator(
            output_path,
            ignore_dataset_type=args.path_ignore_dataset_type,
            ignore_dataset_name=args.path_ignore_dataset_name)
            

class ClassificationOutputRecordGenerator(OutputRecordGenerator):
    def __init__(self, args) -> None:
        self.path_creator = ClassificationOutputRecordGenerator.__prepare_path_creator(args)
        self.image_writer = OutputRecordGenerator.prepare_image_writer(args)

    def generate_output_record(self, data: Tuple[Hashable, pd.Series], type: str) -> None:
        (_, record) = data
        dataset_name = record['dataset']
        frame_path = record['frame_path']
        dest_frame_name = record['proposed_name']
        masks_data = record['mask_data']

        for mask_data in masks_data:
            dest_frame_path = self.path_creator.create_frame_path(dataset_type=type, dataset_name=dataset_name, class_name=mask_data.class_name, file_name=dest_frame_name)
            self.image_writer.write_frame(frame_path, dest_frame_path)


    @staticmethod
    def __prepare_path_creator(args) -> SegmentationPathCreator:
        clean_output = args.force
        output_path = args.output_path
        if clean_output:
            OutputRecordGenerator.clean_output_dir(output_path)

        return ClassificationPathCreator(
            output_path,
            ignore_dataset_type=args.path_ignore_dataset_type,
            ignore_dataset_name=args.path_ignore_dataset_name)
