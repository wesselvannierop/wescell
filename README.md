# Unsupervised Cell Segmentation by Invariant Information Clustering

## Introduction

Unsupervised Cell Segmentation by Invariant Information Clustering created by Wessel L. van Nierop.
Implemented using PyTorch and PyTorch Lightning.

## Getting Started

Setup the environment with the following commands:

```bash
conda create --name wescell python=3.9
conda activate wescell
pip install -r requirements-pytorch.txt
pip install -r requirements.txt
```

## Train new model

Use the config yml given below and run the code using:

```bash
python integration_train.py
```

It will automatically use the GPU when available.

By default, the code logs loss and metrics to Tensorboard.

```bash
tensorboard --logdir="PATH/TO/checkpoint_dir"
```

### Train YAML

```yaml
num_workers: 16 # Set to 0 for development
seed: 321
float32_matmul_precision: "highest" # Optionally, use 'high' or 'medium' for increased speed.

ex_name: "test"

data:
  evican_dir: null # Path to EVICAN dataset. Mutex with data_dir.
  data_dir: "/PATH/TO/DATASET/SPLITS" # Path to dataset with atleast subfolder "train". Images should be .jpg. Mutex with evican_dir.
  labeled_data_dir: "/PATH/TO/validation_data/splits/" # This can be any folder containing the subfolder "val" with images as .jpg and masks as .png
  crop_size: 512
  jitter_strength: 1.0 # for augmentation
  blur_strength: 1.0 # for augmentation
  rotations: [0, -90, 90, 180]
  val_crop: 2072
  illumination_correction:
    downscale_gb_factor: 8
    gauss_size: 31
    sigma: 75
  normalization: [0.4465, 0.1592] # Computed using EVICAN

train:
  val_check_interval: 22 # Validation interval in batches
  nr_of_clusters: 2 # Number of classes the model should try to identify (note that most of the code was written for 2)
  consider_neighbouring_pixels: 1 # How many neighbouring pixels the model should optimize for.
  entropy_coeff: 1.0 # How much weight to add to maximizing marginal entropy (equal class predictions)
  lr: 0.0005
  batch_size: 24
  val_batch_size: 1 # Note that the images are of size val_crop
  max_steps: 2565 # Training stops after this amount of batches
  optimizer: 'adamw'
  fast_dev_run: False # for debugging
  save_checkpoint_every_n_epochs: 1
  save_top_k: 10 # save top 10 best checkpoints
  checkpoint_dir: "PATH/TO/checkpoint_dir" # point to a folder where your experiments will be saved.
  data_loader_kwargs:
    pin_memory: False
    persistent_workers: True
```

The `evican_dir` points to the EVICAN dataset folder. Make sure `data_dir` is set to null.
If you wish to use another dataset. Make sure `evican_dir` is set to null and set the `data_dir` folder to a folder with a subfolder "train". The images should be .jpg.

The EVICAN folder needs a subfolder with `Images/EVICAN_train2019`.
It will automatically ignore the Background images that are included in the EVICAN dataset.
The dataloader is also able to load the partially annotated masks from the `Masks/EVICAN_train_masks` folder. But this is disabled for the unsupervised model.

The `labeled_data_dir` points to the `validation_data/splits` dataset folder.
This loads the images from the corresponding split and expects the images to be `.jpg` and the masks to be `.png`.

The `num_workers` threads for data loading. Set this to 0 for development. This also requires `persistent_workers` to be set to False.
The default config as given above is suitable for the compute server (NVIDIA RTX 3090 GPU with 24 GB VRAM, i9 11900K CPU with 64 GB RAM).

If you change the `batch_size`, note that validation is set to occur every `val_check_interval` batches and training will be stopped after `max_steps` batches.

## Validation of a previously trained model

Use the config yml given below and run the code using:

```bash
python integration_validation.py
```

### Validation only YAML

```yaml
ckpt_path: "weights/wescell.ckpt"
val_dir: "PATH/TO/validation_data/splits/"
normalization: [0.4465, 0.1592] # EVICAN
batch_size: 1
device: "cuda"
num_workers: 8
mode: "val" # val/test
save_folder: null

# Preprocessing
illumination_correction:
  downscale_gb_factor: 8
  gauss_size: 31
  sigma: 75
  disable: False
```
