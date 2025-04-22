# Dataset Directory Structure

The dataset for this project is organized in the following structure:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── annotation/
│   ├── image1.xml
│   ├── image2.xml
│   └── ...
│
├── converted_labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
|
├── train/
│   ├── images/
│   ├── labels/
│
├── val/
│   ├── images/
│   ├── labels/
│
└── test/
    ├── images/
    ├── labels/
```

## Description of Folders

- **images/**: Contains all the raw images used in the dataset.
- **annotation/**: Contains annotation files corresponding to each image. These files provide metadata or labels for the images.
- **converted_labels/**: Contains the annotations in the format suitable for YOLO 
- **train/**: Contains the training dataset, further divided into:
  - `images/`: Images used for training.
  - `labels/`: Labels corresponding to the training images.
- **val/**: Contains the validation dataset, further divided into:
  - `images/`: Images used for validation.
  - `labels/`: Labels corresponding to the validation images.
- **test/**: Contains the test dataset, further divided into:
  - `images/`: Images used for testing.
  - `labels/`: Labels corresponding to the test images.

## Note

- Ensure that the directory structure is maintained as shown above for the proper functioning of the project.
- The `annotation/` folder should have a one-to-one correspondence between images and annotation files.
- The `train/`, `val/`, and `test/` folders should be populated with their respective subsets of the dataset.
- The script to convert annotations is at `/AI/convert_annotations.py`
- The script to train test split is at `/AI/train_test_split.py`
- Dataset downloaded from: https://www.kaggle.com/datasets/dataclusterlabs/indian-signboard-image-dataset?resource=download
