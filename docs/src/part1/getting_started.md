# Getting Started

## Installation

Prerequirements:
- Conda
- Make
- CMake

```bash
conda create -f cu124.yaml
conda activate cu124
```

Requirements: 
```bash
make setup_python
make setup_detrex
```

If c++ CUDA runtime
```bash
make download_and_build_torchtrt
```

Build `benchmark.cpp`:

```bash
make build_cpp
make compile_cpp
```


If error crypt.h
```
export CPATH=$(CONDA_PREFIX)/include
```