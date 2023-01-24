# Dataset Preparation Script

This script is designed to prepare dataset for AI training.
The script is written in Python 3.8 and makes use of several popular libraries, including Pandas, Numpy and scikit-learn.

## Getting Started

To use the script, first ensure that you have the necessary libraries installed.
The script requires Pandas, Numpy, PyYAML, Pillow and scikit-learn.
It is recommended to create a virtual environment(venv) before installing the dependencies.
You can create a venv by running the following command:

```
python -m venv myenv
```

Activate the virtual environment by running the command

- For Linux/MacOS
  ```
  source myenv/bin/activate
  ```
- For Windows
  ```
  myenv\Scripts\activate
  ```
  You can install the required libraries using requirements.txt

```
pip install -r requirements.txt
```

## Input Data

The script takes a path to a folder containing images as input.
The folder should contain the images that you wish to prepare for dataset split into test, train and validation sets.

### How to get Hyperkvasir

To preview dataset structure and download custom data visit: https://osf.io/mh9sj/  
Whole dataset can be download here from prepared links: https://datasets.simula.no/hyper-kvasir/
| File | Description | Size | Download |
|-----------------------------------|-------------------------------------------------|--------|-------------------------------------------------------------------------------------|
| hyper-kvasir.zip | The entire HyperKvasir dataset in one zip file. | 58.6GB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir.zip |
| hyper-kvasir-labeled-images.zip | The labeled image part of HyperKvasir. | 3.9GB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip |
| hyper-kvasir-labeled-videos.zip | The labeled video part of HyperKvasir. | 25.2GB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip |
| hyper-kvasir-segmentation.zip | The segmentation part of HyperKvasir. | 46MB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip |
| hyper-kvasir-unlabeled-images.zip | The unlabeled image part of HyperKvasir. | 29.4GB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-unlabeled-images.zip |

### How to get ERS

> The dataset is available for free for research purposes.  
> To get access, contact the team: <cvlab@eti.pg.edu.pl> or <jan.cychnerski@eti.pg.edu.pl>  
> Visit website for more information: https://cvlab.eti.pg.gda.pl/en/publications/endoscopy-dataset

## Output Data

The script produces three outputs, a training set, a test set and a validation set. All sets are saved in a specified directory.

## Running the Script

To run the script, use the following command:

```
python src\dataset_preparation\main.py
               [-h, --help]
               [--train-size TRAIN_SIZE]
               [--test-size TEST_SIZE]
               [--validation-size VALIDATION_SIZE]
               [--path-ignore-dataset-type PATH_IGNORE_DATASET_TYPE]
               [--path-ignore-dataset-name PATH_IGNORE_DATASET_NAME]
               [--path-ignore-img-type PATH_IGNORE_IMG_TYPE]
               [--path-ignore-class-name PATH_IGNORE_CLASS_NAME]
               [--output-path OUTPUT_PATH]
               [-f, --force]
               [--copy-strategy {duplicate,symlink}]
               [--img-mode IMG_MODE]
               [--mask-mode MASK_MODE]
               [--training-type {segmentation,classification}]
               [--binary]
               [--hyperkvasir-path HYPERKVASIR_PATH]
               [--ers-path ERS_PATH]
               [--ers-use-seq]
               [--ers-use-empty-masks]
               [--ers-class-mapper-path ERS_CLASS_MAPPER_PATH]
```

Replace "HYPERKVASIR_PATH" and “ERS_PATH” with the path to your input images and "OUTPUT_PATH" with the path where you want to save the output sets. Use the optional arguments to specify the type of training, size of train, test and validation sets and various other options.

More information presented in help option after running below command from project root

```
python src\dataset_preparation\main.py --help
```

-   `-h`, `--help`  
show this help message and exit  
- `--train-size TRAIN_SIZE`  
Size of train set  
- `--test-size TEST_SIZE`  
Size of test set  
- `--validation-size VALIDATION_SIZE`  
Size of validation set  
- `--path-ignore-dataset-type PATH_IGNORE_DATASET_TYPE`  
Flag specyfing whether the output path should contain dataset-type (train/test/validation)  
- `--path-ignore-dataset-name PATH_IGNORE_DATASET_NAME`  
Flag specyfing whether the output path should ignore dataset name (examples: hyperkvasir/ers)  
- `--path-ignore-img-type PATH_IGNORE_IMG_TYPE`  
Flag specyfing whether the output path should contain type of image (images/masks)  
- `--path-ignore-class-name PATH_IGNORE_CLASS_NAME`  
Flag specyfing whether the output path should contain class name (polyp/ulcer)  
- `--output-path OUTPUT_PATH`  
Output path for generated data (should be empty)  
- `-f`, `--force`  
Clears output-path if anything exists  
- `--copy-strategy {duplicate,symlink}`  
Strategy used when copying unmodified files to ouput dir  
- `--img-mode IMG_MODE`  
Output image mode compatible with PIL.  
Examples are 'L' for grayscale, RGB, RGBA.  
If not selected then image will be copied as is.  
- `--mask-mode MASK_MODE`  
Output mask image mode compatible with PIL.  
Examples are 'L' for grayscale, RGB, RGBA.  
If not selected then mask will be copied as is.  
- `--training-type {segmentation,classification}`  
Type of training  
- `--binary`  
Flag specifying whether the segmentation should be binary.  
This means no classes, just masks.  
Only classes mapped in class mapper will be used.  
- `--hyperkvasir-path HYPERKVASIR_PATH`  
Path for HyperKvasir dataset (contains folders "labeled-images" and "segmented-images")  
- `--ers-path ERS_PATH`  
Path for ERS dataset (folder containing patients ids e.g. "0001")  
- `--ers-use-seq`  
Use sequences for ERS dataset (e.g. "seq_01")  
- `--ers-use-empty-masks`  
Flag specifying whether images with empty masks should be used for segmentation  
- `--ers-class-mapper-path ERS_CLASS_MAPPER_PATH`  
Localization of class mapper yaml file.
Mapping is done only for ers dataset.  
Records with keys that are not mapped in the file will be skipped.   
Mappers directory contains mapping files for 2, 5 and 10 class problems (2-class.yaml, 5-class.yaml, 10-class.yaml)


### Additional information
hyperkvasir dataset does not provide patient id. It is highly likely that samples from one patient will be split accross more than one of: train, test, or validation datasets.  
ERS multi label images are copied multiple times  the number of copies is equal to the number of classes.  
Output path can be configured as follows: (output-path)/(dataset-type)/(dataset-name)/(img-type)/(class-name)  