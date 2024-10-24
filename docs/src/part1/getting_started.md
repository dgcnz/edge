# Getting Started

TODO: 
- [ ] Introduce project structure 

## Installation

TODO: 
- [ ] Write installation instructions nicely

Prerequirements:
- Conda
- Make
- CMake

Create conda environment:
```bash
conda create -f cu124.yaml
conda activate cu124
```

Install python requirements: 
```bash
make setup_python
make setup_detrex
```

If you need the C++ runtime with TensoRT:
```bash
make download_and_build_torchtrt
# To build the `benchmark` executable
make build_cpp
make compile_cpp
```

## Downloading datasets

If you have a designated folder for datasets, use it, for the purpose of this tutorial, we'll use `~/datasets`:

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

## Setting up environment variables

```bash
# Necessary to avoid TorchInductor and ModelOpt errors looking for crypt.h
export CPATH=$(CONDA_PREFIX)/include  
# To help detectron2 locate the dataset, set to your local path containing COCO
export DETECTRON2_DATASETS=$DATASETS
```