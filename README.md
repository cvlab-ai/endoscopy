# Dataset Preparation Script

This script is designed to prepare ready to train endoscopy dataset for neural network purposes.
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

The script supports processing of 2 datasets: Hyperkvasir and ERS. It takes paths to these datasets, processes them and copies their content into output directory splitting it into test train and validation sets.

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
               [--path-ignore-dataset-type]
               [--path-ignore-dataset-name]
               [--output-path OUTPUT_PATH]
               [-f, --force]
               [--copy-strategy {duplicate,symlink}]
               [--img-mode IMG_MODE]
               [--mask-mode MASK_MODE]
               [--training-type {binary-seg,multilabel-seg,multilabel-classification}]
               [--hyperkvasir-path HYPERKVASIR_PATH]
               [--ers-path ERS_PATH]
               [--ers-use-seq]
               [--ers-use-empty-masks]
               [--ers-class-mapper-path ERS_CLASS_MAPPER_PATH]
```

Replace `HYPERKVASIR_PATH` and `ERS_PATH` with the path to your input images and `OUTPUT_PATH` with the path where you want to save the output sets. Use the optional arguments to specify the type of training, size of train, test and validation sets and various other options.

Sample command to prepare data for [FCBFormer](https://github.com/ESandML/FCBFormer):
```
python3 main.py --training-type binary-seg --ers-path "/raid/gwo/public/gastro/ers" --output-path "./fcb-ers" -f --train-size 1.0 --test-size 0.0 --ers-class-mapper-path "mappers/2-class-polyp.yaml" --ers-use-seq --mask-mode "RGB" --path-ignore-dataset-name --path-ignore-dataset-type
```
Sample command to prepare data for [ESFPNet](https://github.com/dumyCq/ESFPNet):
```
python3 main.py --training-type binary-seg --ers-path "/raid/gwo/public/gastro/ers" -f --train-size=0.8 --test-size=0.2 --ers-class-mapper-path "./mappers/2-class-polyp.yaml" --ers-use-seq 
```

More information is presented in help option after running below command from project root  
```
python3 main.py --help
```

-   `-h`, `--help`  
show the help message and exit  
- `--train-size TRAIN_SIZE`  
Size of train set split (sum of train, test, validation size must equal 1.0). Defaults to 0.7.
- `--test-size TEST_SIZE`  
Size of test set split (sum of train, test, validation size must equal 1.0). Defaults to 0.1.
- `--validation-size VALIDATION_SIZE`  
Size of validation set split (sum of train, test, validation size must equal 1.0). Defaults to 0.2.
- `--path-ignore-dataset-type`  
Flag specifying whether the output path should contain dataset-type (train/test/validation)  
e.g. for multilabel-seg with a flag → `ers/masks/polyp/1.png`  
without flag → `test/ers/masks/polyp/1.png`  
Useful when there is no need to split data into subsets.
- `--path-ignore-dataset-name`  
Flag specifying whether the output path should ignore dataset name (examples: hyperkvasir/ers)  
e.g. for multilabel-seg with a flag → `test/masks/polyp/1.png`  
without flag → `test/ers/masks/polyp/1.png`
- `--output-path OUTPUT_PATH`  
Output path for generated data (path content should be empty, no folders nor files inside, otherwise use -f to force clear). In general, output directory will generate the following structure: `(output-path)/(dataset-type)/(dataset-name)/(images|masks)/(class-name)` (e.g. `home/train/ERS/masks/polyp`), but the behaviour can be modified by `path-ignore-*` flags. Defaults to current working directory.
- `-f`, `--force`  
Clears output-path if anything exists  
- `--copy-strategy {duplicate,symlink}`  
Strategy used when copying unmodified files to output dir. Defaults to duplicate on Windows and symlink on other platforms.
- `--img-mode IMG_MODE`  
Output image mode compatible with PIL.  
Examples are `L` for grayscale, `RGB`, `RGBA`.  
If not specified then image will be copied as is.  
- `--mask-mode MASK_MODE`  
Output mask image mode compatible with PIL.  
Examples are `L` for grayscale, `RGB`, `RGBA`.  
If not specified then mask will be copied as is.  
- `--training-type {binary-seg,multilabel-seg,multilabel-classification}`  
Argument is required!  
Type of training. When set to `multilabel-classification`:
    - no masks are copied to the output directory.
    - ers-use-empty-masks parameter will be overridden to true.

- `--binary`  
Flag specifying whether the segmentation should be binary. Useful for 2 class segmentation problems like disease and normal. In this mode, there will be no color reversing in classes labeled as healthy in ERS class mapping. Defaults to false. See [healthy flag section](#healthy-flag).
- `--hyperkvasir-path HYPERKVASIR_PATH`  
Path for Hyperkvasir dataset (must contain folders `labeled-images` and `segmented-images`)  
- `--ers-path ERS_PATH`  
Path for ERS dataset (must contain patient id directories e.g. `0001`)  
- `--ers-use-seq`  
Use sequences directory for ERS dataset (e.g. "seq_01"). Defaults to false.
- `--ers-use-empty-masks` 
Flag specifying whether images with empty mask files should be used for segmentation. Independently, script will use empty mask files that belong to healthy classes. See [healthy flag section](#healthy-flag). Defaults to false. For training type `multilabel-classification` it is overridden to true.
- `--ers-class-mapper-path ERS_CLASS_MAPPER_PATH`  
Localization of class mapper yaml file. Mapping is done only for ers dataset. See [class mapping section](###class-mapping). Mappers directory contains sample mapping files ready for 2, 5 and 10 class problems (2-class.yaml, 5-class.yaml, 10-class.yaml).

## Data processing

### Class mapping

Mappings for classes are defined in `.yaml` files. General structure of the file is as follows:

```
output_class_c:
  classes:
	- c01
	- c02
	- ...
output_class_q:
  classes:
    - q01
    - q02
    - ...
  positive: true # Optional, defaults to false
...
```

This means that classes `c01` and `c01` from ERS would be mapped to `output_class_c`. What's more, if there was another definition, like:

```
disease:
  classes:
    - c01
```

then the `c01` would be mapped to both `output_class_c` and `disease`. This scheme allows many-to-many mappings.

Records assigned to class that are not mapped will be dropped, therefore will not exist in the output dataset. Default mapping behavior, that occurs when class mapper is unspecified, maps classes one to one, ex: `c01 -> c01` and assumes that `h01-h07` and `b02` are healthy classes.

##### Positive flag
In ERS dataset, some masks are empty files, especially the ones that label healthy classes. By default, such mask files are ignored unless they belong to a class labeled as positive. The script converts them to a valid picture which mode and size is based on the original frame. Filling color of the picture is determined by the following rule:
- if  segmentation is binary **and**, the mask belongs to a none positive class the color is black.
- otherwise, the color is white.

If there is a valid mask file that belongs to a class labeled as not positive, and the segmentation is binary, then the colors in the mask will get reversed. White pixels will switch places with the black pixels.

##### Mapping example
Let's assume the following example:

```
ers/0001/
├── frames
│   ├── 00001.png
│   ├── 00002.png
├── labels
│   ├── 00001_c01_c02_q01.png
│   ├── 00001_c03_c04.png
│   ├── 00002_q02.png
```

and mappings as follows:
```
c_class:
  classes:
    - c01
    - c02
    - c03
q_class:
  classes:
    - q01
    - q02
```

The example will be transformed to the following output:
```
.../
├── images
│   ├── 00001.png
│   ├── 00002.png
├── masks
│   ├── c_class
│   │	├── 00001.png //Mask merging 00001_c01_c02_q01.png and 00001_c03_c04.png
│   ├── q_class
│   │	├── 00001.png //Same as 00001_c01_c02_q01.png
│   │	├── 00002.png //Same as 00002_q02.png
```

##### Mask merging
Masks merging takes place when a single frame has multiple masks that map to the same class. The process is as follows: each pixel in the output image is the maximum value of the pixels on corresponding positions of the input images. Example:

![Masks merging example](assets/merging-example.png)

If the masks are to be reversed (see [healthy flag section](#healthy-flag)), then the reversing will take place before merging. Example:

![Reverse masks merging example](assets/merging-reversed-example.png)


## Datasets

### How to get Hyperkvasir

To preview dataset structure and download custom data visit: https://osf.io/mh9sj/  
Whole dataset can be downloaded here from prepared links: https://datasets.simula.no/hyper-kvasir/
| File | Description | Size | Download |
|-----------------------------------|-------------------------------------------------|--------|-------------------------------------------------------------------------------------|
| hyper-kvasir.zip | The entire HyperKvasir dataset in one zip file. | 58.6 GB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir.zip |
| hyper-kvasir-labeled-images.zip | The labeled image part of HyperKvasir. | 3.9 GB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip |
| hyper-kvasir-labeled-videos.zip | The labeled video part of HyperKvasir. | 25.2 GB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip |
| hyper-kvasir-segmentation.zip | The segmentation part of HyperKvasir. | 46 MB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip |
| hyper-kvasir-unlabeled-images.zip | The unlabeled image part of HyperKvasir. | 29.4 GB | https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-unlabeled-images.zip |

### How to get ERS

> The dataset is available for free for research purposes.  
> To get access, contact the team: <cvlab@eti.pg.edu.pl> or <jan.cychnerski@eti.pg.edu.pl>  
> Visit website for more information: https://cvlab.eti.pg.gda.pl/en/publications/endoscopy-dataset

### Additional information
1. Hyperkvasir dataset does not provide patient ids. It is highly likely that samples from one patient will be split across more than one of: train, test, or validation datasets.
2. ERS multi label images are copied multiple times - the number of copies is equal to the number of classes.
