import argparse
import ers_preparator as ers
import hyperkvasir_preparator as hkvs
import data_preparator as data_prep
import pandas as pd
import os
import shutil
import yaml
from enums import TrainingType
from copy_strategy import CopyStrategy, AbstractCopyStrategy
from class_mappers import DummyClassMapper, DictClassMapper, AbstractClassMapper

DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_TEST_SIZE = 0.1
DEFAULT_TRAIN_SIZE = 0.7
EMPTY_FLOAT = -1

def dir_path(path):
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)

def file_path(path):
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)

def is_dir_empty(path):
    return not next(os.scandir(path), None)

def setup_argument_parser():
    epilog = """
        [INFO] hyperkvasir dataset does not provide patient id. It is highly likely that samples from one patient will be split accross more than one of: train, test, or validation datasets.
        [INFO] ERS multi label images are copied multiple times - the number of copies is equal to the number of classes.
    """
    parser = argparse.ArgumentParser(description='Dataset preparator', epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument('--training-type',
                        type=TrainingType,
                        choices=list(TrainingType),
                        default=TrainingType.SEGMENTATION,
                        help="Type of training",
                        required=False)
    parser.add_argument("--train-size",
                        type=float,
                        default=EMPTY_FLOAT,
                        help="Size of train set",
                        required=False)
    parser.add_argument("--test-size",
                        type=float,
                        default=EMPTY_FLOAT,
                        help="Size of test set",
                        required=False)
    parser.add_argument("--validation-size",
                        type=float,
                        default=EMPTY_FLOAT,
                        help="Size of validation set",
                        required=False)
    parser.add_argument("--use-seq",
                        action="store_true",
                        help="Use sequences for ERS dataset (e.g. \"seq_01\")")
    parser.add_argument("--ers-path",
                        help="Path for ERS dataset (folder containing patients ids e.g. \"0001\")",
                        type=dir_path,
                        required=False)
    parser.add_argument("--hyperkvasir-path",
                        help="Path for HyperKvasir dataset (contains folders \"labeled-images\" and \"segmented-images\")",
                        type=dir_path,
                        required=False)
    parser.add_argument("--output-path",
                        help="Output path for generated data (should be empty)",
                        default="./data",
                        type=str,
                        required=False)
    parser.add_argument("-f", "--force",
                        action="store_true",
                        help="Clears output-path if anything exists")
    parser.add_argument("--copy-strategy",
                        help="Strategy used when copying files to ouput dir",
                        default=CopyStrategy.SYMLINK,
                        type=CopyStrategy,
                        choices=list(CopyStrategy),
                        required=False)
    parser.add_argument("--use-empty-masks",
                        action="store_true",
                        help="Flag specifying whether images with empty masks should be used for segmentation")
    parser.add_argument("--ers-class-mapper-path",
                        type=file_path,
                        help="Localization of class mapper yaml file. Mapping is done only for ers dataset. Records with keys that are not mapped in the file will be skipped.",
                        required=False)

    return parser


def main():
    setup_argument_parser()
    args = parse_args()

    prepare_dataset(args.ers_path, args.hyperkvasir_path, args.output_path, args.force, args.use_seq, args.training_type, args.copy_strategy, args.use_empty_masks, args.ers_class_mapper_path, args.train_size, args.validation_size, args.test_size)

def parse_args():
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.hyperkvasir_path is None and args.ers_path is None:
        parser.error("At least one of --hyperkvasir-path and --ers-path required")

    missing_sizes_count = 0
    for arg in [args.train_size, args.test_size, args.validation_size]:
        if arg == EMPTY_FLOAT:
            missing_sizes_count += 1
    if missing_sizes_count == 3:
        args.train_size = DEFAULT_TRAIN_SIZE
        args.test_size = DEFAULT_TEST_SIZE
        args.validation_size = DEFAULT_VALIDATION_SIZE
    elif missing_sizes_count > 1:
        parser.error("Only one of --train-size,--test-size and --validation-size can be skipped")
    if args.train_size == EMPTY_FLOAT:
        args.train_size = 1 - args.test_size - args.validation_size
    if args.test_size == EMPTY_FLOAT:
        args.test_size = 1 - args.train_size - args.validation_size
    if args.validation_size == EMPTY_FLOAT:
        args.validation_size = 1 - args.test_size - args.train_size
    if round(args.train_size + args.test_size + args.validation_size) != 1.0:
        parser.error("Sum of --train-size,--test-size and --validation-size should be equal 1.0")
    if args.force is False and os.path.isdir(args.output_path) and not is_dir_empty(args.output_path):
        parser.error("Output directory should be empty. Use -f to force clean")

    return args


def prepare_dataset(ers_path: str, hyperkvasir_path: str, output_path: str, clean_output: bool, use_ers_seq: bool, training_type: TrainingType, copy_strategy: CopyStrategy, use_empty_masks: bool, class_mapper_path: str, train_size: float, val_size: float, test_size: float) -> None:
    if clean_output and os.path.isdir(output_path):
        print(f"Cleaning output dir: {output_path}")
        shutil.rmtree(output_path)
        print("Output path cleaned")

    class_mapper = __create_class_mapper(class_mapper_path)
    use_empty_masks = use_empty_masks or training_type == TrainingType.CLASSIFICATION

    ers_df = ers.generate_dataframe_for_ers(ers_path, class_mapper=class_mapper, use_seq=use_ers_seq, use_empty_masks=use_empty_masks) if ers_path else pd.DataFrame()
    hkvs_df = hkvs.generate_dataframe_for_hyperkvasir(hyperkvasir_path, training_type) if hyperkvasir_path else pd.DataFrame()
    result_df = pd.concat([ers_df, hkvs_df])

    train_df, val_df, test_df = data_prep.prepare_data(result_df, train_part=train_size, val_part=val_size)
    copy_mask = training_type == TrainingType.SEGMENTATION
    __fill_output_dir(output_path, 'train', copy_mask, copy_strategy.create(), train_df)
    __fill_output_dir(output_path, 'validation', copy_mask, copy_strategy.create(), val_df)
    __fill_output_dir(output_path, 'test', copy_mask, copy_strategy.create(), test_df)

    print("Dataset prepared")


def __fill_output_dir(root: str, type: str, copy_mask: bool, copy_strategy: AbstractCopyStrategy, df: pd.DataFrame) -> None:
    for loop_index, (row_id, record) in enumerate(df.iterrows()):
        dataset_name = record['dataset']
        class_name = record['class']
        src_img_path = record['img_path']
        img_ext = os.path.splitext(src_img_path)[1]
        dest_img_name = f"{row_id}{img_ext}"

        dst_img_path = os.path.join(root, type, dataset_name, 'images', class_name)
        os.makedirs(dst_img_path, exist_ok=True)
        copy_strategy.copy(src_img_path, os.path.join(dst_img_path, dest_img_name))
        
        if copy_mask:
            mask_dir_path = os.path.join(root, type, dataset_name, 'masks', class_name)
            os.makedirs(mask_dir_path, exist_ok=True)
            copy_strategy.copy(record['mask_path'], os.path.join(mask_dir_path, dest_img_name))
        
        if loop_index % 1000 == 0:
            print(f"Processed {loop_index} images")
            
    print(f"Processed all images from {type} dataset")

def __create_class_mapper(class_mapper_path: str) -> AbstractClassMapper: 
    if class_mapper_path is None:
        return DummyClassMapper()
    with open(class_mapper_path, "r") as stream:
        return DictClassMapper(yaml.safe_load(stream))

if __name__ == '__main__':
    main()
