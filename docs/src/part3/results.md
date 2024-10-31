# Benchmarks and Results

## Running the benchmarks

Download the model:
```bash
!wget https://huggingface.co/dgcnz/dinov2_vitdet_DINO_12ep/resolve/main/model_final.pth -O artifacts/model_final.pth ⁠
```

Before running the benchmarks make sure you have compiled your desired model. 
```bash
python -m scripts.export_tensorrt --config-name dinov2 amp_dtype=fp32 trt.enabled_precisions="[fp32, bf16, fp16]" 
# ...
# OUTPUT DIR: outputs/2024-10-31/10-43-31
```

The outputs of this script will be found in the directory specified by `OUTPUT DIR`. The directory will contain the following files:

```
├── export_tensorrt.log     # log file
├── .hydra
│   ├── config.yaml         # config file
│   ├── hydra.yaml
│   └── overrides.yaml      
├── model.ts                # compiled torchscript model
└── predictions.png         # sample predictions for the model
```

There are three possible runtimes to benchmark, examples of how to run the benchmarks are shown below:

**Python Runtime, no TensorRT**
```bash
python -m scripts.benchmark_gpu compile_run_path=outputs/2024-10-31/10-43-31 n_iter=100 load_ts=False amp_dtype=fp16
```

**Python Runtime with TensorRT**
```bash
python -m scripts.benchmark_gpu compile_run_path=outputs/2024-10-31/10-43-31 n_iter=100 load_ts=True
```

**C++ Runtime with TensorRT**
```bash
./build/benchmark --model outputs/2024-10-31/10-43-31/model.ts --n_iter=100
```

## Results


**Python Runtime, no TensorRT**

| model's precision | amp_dtype              | latency (ms)   |
| ----------------- | ---------------------- | -------------- |
| fp32              | fp32+fp16              | 66.322 ± 0.927 |
| fp32              | fp32+bf16              | 66.497 ± 1.052 |
| fp32              | fp32                   | 76.275 ± 0.587 |

Max memory usage for all configurations is ~1GB.

**Python Runtime, with TensorRT**

| model's precision | trt.enabled_precisions | latency (ms)   |
| ----------------- | ---------------------- | -------------- |
| fp32+fp16         | fp32+bf16+fp16         | 15.369 ± 0.023 |
| fp32              | fp32+bf16+fp16         | 23.164 ± 0.031 |
| fp32              | fp32+bf16              | 25.148 ± 0.030 |
| fp32              | fp32                   | 38.381 ± 0.022 |

Max memory usage for all configurations is ~500MB except for fp32+fp32 which is ~770MB.

**C++ Runtime, no TensorRT**

| model's precision | trt.enabled_precisions | latency (ms)   |
| ----------------- | ---------------------- | -------------- |
| fp32+fp16         | fp32+bf16+fp16         | 15.433 ± 0.029 |
| fp32              | fp32+bf16+fp16         | 23.263 ± 0.027 |
| fp32              | fp32+bf16              | 25.255 ± 0.014 |
| fp32              | fp32                   | 38.465 ± 0.029 |


Max memory usage for all configurations is ~500MB except for fp32+fp32 which is ~770MB.

---

Note: For some reason in the latest version of torch_tensorrt, `bfloat16` precision is not working well and it's not achieving the previously measured performance of (13-14ms) and/or failing compilation. 

We include the previous results for completeness, in case the issue is resolved in the future.

| Runtime | model's precision | trt.enabled_precisions | latency | memory (mb) |
| ------- | ----------------- | ---------------------- | ------- | ----------- |
| cpp+trt | fp32              | fp32+fp16              | 13.984  | 500         |
| cpp+trt | fp32              | fp32+bf16+fp16         | 13.898  | 500         |
| cpp+trt | fp32              | fp32+bf16              | 17.261  | 500         |
| cpp+trt | bf16              | fp32+bf16              | 22.913  | 500         |
| cpp+trt | bf16              | bf16                   | 22.938  | 500         |
| cpp+trt | fp32              | fp32                   | 37.639  | 770         |


