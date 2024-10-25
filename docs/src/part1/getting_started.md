# Getting Started

```{contents}
```

## Project structure

The project is structured as follows:

```
.
├── artifacts           # Model weights and scripts I/O
├── build               # Build directory (location for cpp executables)
├── cpp                 # source code for cpp executables
├── detrex              # fork of detrex
├── docs                # documentation
├── logs                
├── notebooks           # jupyter notebooks
├── output              # [Training] `scripts.train_net` outputs (tensorboard logs, weights, etc)
├── projects            # configurations and model definitions
├── scripts             # utility scripts 
├── src                 # python source code
├── third-party         # third-party c libraries
├── wandb_output        # Output from wandb
├── CMakeLists.txt      # CMake configuration for cpp executables
├── cu124.yaml          # Conda environment file (only system dependencies: cuda, gcc, python)
├── Makefile            # Makefile for project scripts
├── poetry.lock         # Locked python dependencies
├── pyproject.toml      # Poetry configuration
├── README.md 
```

The main folders to focus are `src` and `scripts` as it is where most of the source code lies.

## Installation

First make sure the (bold) pre-requirements are fulfilled:
- **Conda** 
- **Make** 
- CMake (for building cpp executables)


First let's create our conda environment. This will install the cuda runtime and libraries, python, the poetry dependency manager and other stuff:

```bash
conda env create -f cu124.yaml
conda activate cu124
```

To avoid TorchInductor and ModelOpt errors looking for `crypt.h`:

```bash
conda env config vars set CPATH=$CONDA_PREFIX/include  
conda activate cu124
```

Installing the dependencies requires some manual building (`detrex`, `detectron2`), so we can use the make commands to do it for us:

```bash
make setup_python
make setup_detrex
```

(Optional) If you need the C++ TensorRT runtime and the accompanying benchmark executables, you can build them with the following commands:

```bash
make download_and_build_torchtrt
# To build the `benchmark` executable
make build_cpp
make compile_cpp
```

This will automatically download the necessary files and build the libraries for you.


## Downloading datasets (training-only)

If you have a designated folder for datasets, use it, for the purpose of this tutorial, we'll use `~/datasets`. We'll test with the COCO dataset, so let's download it:

```bash
cd ~/datasets
mkdir coco
cd coco

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip

unzip annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
```

To point the `detectron2` library to the dataset directory, we need to set the `DETECTRON2_DATASETS` environment variable:

```bash
conda env config vars set DETECTRON2_DATASETS=~/datasets
conda activate cu124
```