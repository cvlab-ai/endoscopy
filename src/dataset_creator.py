import pandas as pd
from src.training_type import TrainingType
from src.ers_preparator import ErsPreparator
from src.hyperkvasir_preparator import HyperkvasirPreparator
from src.splitter import DataSplitter
from src.output_record_generator import SegmentationOutputRecordGenerator, ClassificationOutputRecordGenerator, OutputRecordGenerator


class DatasetCreator:
    def __init__(self, args) -> None:
        self.output_record_generator = DatasetCreator.__prepare_record_generator(args)
        self.data_splitter = DatasetCreator.__prepare_data_splitter(args)

        self.ers_preparator = ErsPreparator(args)
        self.hkvs_preparator = HyperkvasirPreparator(args)


    def create(self) -> None:
        df = self.__generate_dataframes()
        train_df, val_df, test_df = self.data_splitter.split_and_prepare(df)

        print(f"Data of size {df.shape[0]} split to sizes: \n train_size={train_df.shape[0]} \n validation_size={val_df.shape[0]} \n test_size={test_df.shape[0]}")

        self.__fill_output_dir(train_df, 'train')
        self.__fill_output_dir(val_df, 'validation')
        self.__fill_output_dir(test_df, 'test')

        print("Dataset prepared")

    def __fill_output_dir(self, df: pd.DataFrame, type: str):
        print(f"Processing images from {type} dataset")

        for loop_index, data in enumerate(df.iterrows()):
            self.output_record_generator.generate_output_record(data, type)
        
            if loop_index % 100 == 0 and loop_index > 0:
                print(f"Processed {loop_index} images")
            
        print(f"Processed all images from {type} dataset")

    
    def __generate_dataframes(self) -> pd.DataFrame:
        return pd.concat([
            self.ers_preparator.generate_dataframe(),
            self.hkvs_preparator.generate_dataframe()])

    
    @staticmethod
    def __prepare_record_generator(args) -> OutputRecordGenerator:
        if args.training_type == TrainingType.MULTILABEL_CLASSIFICATION:
            return ClassificationOutputRecordGenerator(args)
        else:
            return SegmentationOutputRecordGenerator(args)
    
    @staticmethod
    def __prepare_data_splitter(args) -> DataSplitter:
        return DataSplitter(
            train_part=args.train_size,
            val_part=args.validation_size)