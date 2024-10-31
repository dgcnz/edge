# Benchmarks and Results

```{contents}
```

## Running the benchmarks

Before running the benchmarks, make sure you have downloaded the trained model (see {ref}`part1:downloadmodel`) and compiled it (see {ref}`part2:compilingmodel`).

We'll assume that the output directory of `export_tensorrt`'compilation was `outputs/2024-10-31/10-43-31`.

There are three possible runtimes to benchmark, examples are shown below:

**Python Runtime, no TensorRT**

This mode takes the uncompiled model and runs it with mixed precision (fp16 or bf16) or full precision (fp32).

```bash
python -m scripts.benchmark_gpu compile_run_path=outputs/2024-10-31/10-43-31 n_iter=100 load_ts=False amp_dtype=fp16
```

**Python Runtime with TensorRT**

```bash
python -m scripts.benchmark_gpu compile_run_path=outputs/2024-10-31/10-43-31 n_iter=100 load_ts=True
```

**C++ Runtime with TensorRT**

Make sure you have built the C++ runtime (see {ref}`part1:installation`).
```bash
./build/benchmark --model outputs/2024-10-31/10-43-31/model.ts --n_iter=100
```

## Results

Benchmarking was done on a NVIDIA RTX 4060 Ti GPU with 16GB of VRAM. Results are shown below.

```{table} **Python Runtime, no TensorRT**
:name: py_notrt

| model's precision | amp_dtype              | latency (ms)   |
| ----------------- | ---------------------- | -------------- |
| fp32              | fp32+fp16              | 66.322 ± 0.927 |
| fp32              | fp32+bf16              | 66.497 ± 1.052 |
| fp32              | fp32                   | 76.275 ± 0.587 |

```

Max memory usage for all configurations is ~1GB.


```{table} **Python Runtime, with TensorRT**
:name: py_trt

| model's precision | trt.enabled_precisions | latency (ms)   |
| ----------------- | ---------------------- | -------------- |
| fp32+fp16         | fp32+bf16+fp16         | 15.369 ± 0.023 |
| fp32              | fp32+bf16+fp16         | 23.164 ± 0.031 |
| fp32              | fp32+bf16              | 25.148 ± 0.030 |
| fp32              | fp32                   | 38.381 ± 0.022 |

```
Max memory usage for all configurations is ~500MB except for fp32+fp32 which is ~770MB.

 
```{table} **C++ Runtime, with TensorRT**
:name: cpp_trt

| model's precision | trt.enabled_precisions | latency (ms)   |
| ----------------- | ---------------------- | -------------- |
| fp32+fp16         | fp32+bf16+fp16         | 15.433 ± 0.029 |
| fp32              | fp32+bf16+fp16         | 23.263 ± 0.027 |
| fp32              | fp32+bf16              | 25.255 ± 0.014 |
| fp32              | fp32                   | 38.465 ± 0.029 |
```

Max memory usage for all configurations is ~500MB except for fp32+fp32 which is ~770MB.

:::{note}

For some unknown reason, `bfloat16` precision is not working well and it's not achieving the previously measured performance of (13-14ms) and/or failing compilation in the latest version of `torch_tensorrt`.

We include the previous results for completeness, in case the issue is resolved in the future.

```{table} **C++ Runtime, with TensorRT (previous results)**
:name: cpp_trt_old

| model's precision | trt.enabled_precisions | latency |
| ----------------- | ---------------------- | ------- |
| fp32              | fp32+fp16              | 13.984  |
| fp32              | fp32+bf16+fp16         | 13.898  |
| fp32              | fp32+bf16              | 17.261  |
| bf16              | fp32+bf16              | 22.913  |
| bf16              | bf16                   | 22.938  |
| fp32              | fp32                   | 37.639  |
```

:::
