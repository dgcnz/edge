VENV           = .venv
VENV_PYTHON    = $(VENV)/bin/python
# make it work on windows too
ifeq ($(OS), Windows_NT)
    VENV_PYTHON=$(VENV)/Scripts/python
endif
SYSTEM_PYTHON  = $(or $(shell which python3), $(shell which python))
PYTHON = $(VENV_PYTHON)

help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

clean-logs: ## Clean logs
	rm -rf logs/**

format: ## Run pre-commit hooks
	pre-commit run -a

sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main

test: ## Run not slow tests
	pytest -k "not slow"

test-full: ## Run all tests
	pytest

train: ## Train the model
	python src/train.py


demo_vitdet:
	cd detrex && \
	python demo/demo.py --config-file detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_b_100ep.py \
						--input ../artifacts/idea.jpg  \
						--output ../artifacts/demo_output.jpg \
						--opts train.init_checkpoint=../artifacts/model_final_435fa9.pkl train.device=cpu 

demo_eva:
	cd detrex && \
	python demo/demo.py --config-file projects/dino_eva/configs/dino-eva-02/dino_eva_02_vitdet_b_4attn_1024_lrd0p7_4scale_12ep.py \
													--input ../artifacts/idea.jpg  \
													--output ../artifacts/demo_output_eva.jpg \
													--opts train.init_checkpoint=../artifacts/dino_eva_02_in21k_pretrain_vitdet_b_4attn_1024_lrd0p7_4scale_12ep.pth train.device=cpu model.device=cpu  


demo_vitdet_dino:
	cd detrex && \
	python demo/demo.py --config-file projects/dino/configs/dino-vitdet/dino_vitdet_base_4scale_12ep.py \
													--input ../artifacts/idea.jpg \
													--output ../artifacts/demo_output_dino_vit.jpg \
													--opts train.init_checkpoint=../artifacts/dino_vitdet_base_4scale_50ep.pth train.device=cpu model.device=cpu

module_avail: ## greppable module avail
	module -t avail 2>&1

load_modules:
	module load 2023
	module load Python/3.11.3-GCCcore-12.3.0

unload_modlues:
	module unload Python/3.11.3-GCCcore-12.3.0
	module unload 2023

setup_env: # setup the virtual environment and download dependencies
	$(SYSTEM_PYTHON) -m venv .venv
	$(PYTHON) -m pip install poetry
	$(PYTHON) -m poetry install

scat: ## cat slurm log with param
	cat scripts/slurm_logs/slurm_output_$(id).out

download_torch_cpp:
	cd third-party && wget https://download.pytorch.org/libtorch/nightly/cu124/libtorch-cxx11-abi-shared-with-deps-latest.zip

# pip3 install --pre torch torchvision tensorrt torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu124
#
# poetry export --without-hashes --format=requirements.txt > requirements.txt
#
# python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)
#
#

build_cpp:
	cmake -DCMAKE_CXX_FLAGS=-O3 -Bbuild

compile_cpp:
	cmake --build build --config Release

# We need to include crypt.h from $CONDA_PREFIX/include
test_modelopt_installation:
	CPATH=$(CONDA_PREFIX)/include python -c "import modelopt.torch.quantization.extensions as ext; print(ext.cuda_ext); print(ext.cuda_ext_fp8)"