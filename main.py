import argparse
import os
import sys

from src.training_type import TrainingType
from src.copy_strategy import CopyStrategy
from src.dataset_creator import DatasetCreator

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
        [INFO] hyperkvasir dataset does not provide patient id. It is highly likely that samples from one patient will be split across more than one of: train, test, or validation datasets.
        [INFO] ERS multi label images are copied multiple times - the number of copies is equal to the number of classes.
        [INFO] Output path can be configured as follows: (output-path)/(dataset-type)/(dataset-name)/(img-type)/(class-name) e.g. home/train/ERS/image/polyp
    """
    parser = argparse.ArgumentParser(description='Dataset preparator', epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    #Dataset size:
    parser.add_argument("--train-size",
                        type=float,
                        default=EMPTY_FLOAT,
                        help="Size of train set split (sum of train+test+validation size must equal 1)",
                        required=False)
    parser.add_argument("--test-size",
                        type=float,
                        default=EMPTY_FLOAT,
                        help="Size of test set split (sum of train+test+validation size must equal 1)",
                        required=False)
    parser.add_argument("--validation-size",
                        type=float,
                        default=EMPTY_FLOAT,
                        help="Size of validation set split (sum of train+test+validation size must equal 1)",
                        required=False)

    #Dataset output options
    parser.add_argument("--path-ignore-dataset-type",
                        help="Flag specifying whether the output path should ignore dataset-type (train/test/validation) e.g. with flag -> ers/images/polyp/1.png, without flag -> test/ers/images/polyp/1.png",
                        default=False,
                        action="store_true",
                        required=False)
    parser.add_argument("--path-ignore-dataset-name",
                        help="Flag specifying whether the output path should ignore dataset name (examples: hyperkvasir/ers) e.g. with flag -> test/images/polyp/1.png, without flag -> test/ers/images/polyp/1.png",
                        default=False,
                        action="store_true",
                        required=False)
    parser.add_argument("--output-path",
                        help="Output path for generated data (path content should be empty, no folders nor files inside, otherwise use -f to force clear)",
                        default="./data",
                        type=str,
                        required=False)
    parser.add_argument("-f", "--force",
                        action="store_true",
                        help="Clears output-path if anything exists")
    parser.add_argument("--copy-strategy",
                        help="Strategy used when copying unmodified files to output dir",
                        default=CopyStrategy.DUPLICATE if sys.platform == "win32" else CopyStrategy.SYMLINK,
                        type=CopyStrategy,
                        choices=list(CopyStrategy),
                        required=False)                        

    #Image options
    parser.add_argument('--img-mode',
                        help="Output image mode compatible with PIL. Examples are 'L' for grayscale, RGB, RGBA. If not selected then image will be copied as is.",
                        required=False)
    parser.add_argument('--mask-mode',
                        help="Output mask image mode compatible with PIL. Examples are 'L' for grayscale, RGB, RGBA. If not selected then mask will be copied as is.",
                        required=False)

    #Training specific
    parser.add_argument('--training-type',
                        type=TrainingType,
                        choices=list(TrainingType),
                        help="Type of training",
                        required=True)

    #Hyperkvasir
    parser.add_argument("--hyperkvasir-path",
                        help="Path for HyperKvasir dataset (contains folders \"labeled-images\" and \"segmented-images\")",
                        type=dir_path,
                        required=False)

    #ERS
    parser.add_argument("--ers-path",
                        help="Path for ERS dataset (folder containing patients ids e.g. \"0001\")",
                        type=dir_path,
                        required=False)
    parser.add_argument("--ers-use-seq",
                        action="store_true",
                        help="Use sequences for ERS dataset (e.g. \"seq_01\")")
    parser.add_argument("--ers-use-empty-masks",
                        action="store_true",
                        help="Flag specifying whether images with empty mask files should be used for segmentation. Independently, script will use empty mask files that belong to healthy classes.")
    parser.add_argument("--ers-class-mapper-path",
                        type=file_path,
                        help="Localization of class mapper yaml file. Mapping is done only for ers dataset. Records with keys that are not mapped in the file will be skipped.",
                        required=False)

    return parser


def parse_args(args):
    parser = setup_argument_parser()
    args = parser.parse_args(args)

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
        if args.train_size == 1:
            args.test_size = 0
            args.validation_size = 0
        elif args.test_size == 1:
            args.train_size = 0
            args.validation_size = 0
        elif args.validation_size == 1:
            args.train_size = 0
            args.test_size = 0
        else:
            parser.error("Only one of --train-size,--test-size and --validation-size can be skipped (set one to 1.0 or provide another one)")
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
    if args.ers_use_empty_masks == False and args.training_type == TrainingType.MULTILABEL_CLASSIFICATION:
        args.ers_use_empty_masks = True
        print("[INFO] Ignoring '--ers-use-empty-masks' parameter, since training type is classification")
    if args.copy_strategy == CopyStrategy.SYMLINK:
        print("[INFO] Chosen copy strategy is SYMLINK. Keep in mind that the script may still sometimes create new image files in the output dataset")
    if args.ers_class_mapper_path is None and args.ers_path is not None:
        print("[INFO] No ERS mapper specified. Default behaviour will be used.")
    if args.training_type == TrainingType.MULTILABEL_CLASSIFICATION and args.mask_mode is not None:
        print("[INFO] Ignoring mask-mode parameter for classification training type")
    if args.hyperkvasir_path is not None and args.training_type != TrainingType.BINARY_SEG:
        print("[INFO] It is not recommended to use hyperkvasir feature of this script for non binary segmentation purposes")

    return args


def main(args):
    setup_argument_parser()
    args = parse_args(args)
    DatasetCreator(args).create()


if __name__ == '__main__':
    main(sys.argv[1:])
