import torch
import time
from functools import partial
import contextlib
from src.utils import (
    load_input_fixed,
    TracingAdapter,
)
from src.utils import load_model
from statistics import stdev, mean
import torch_tensorrt
import logging
from pathlib import Path
import detrex
import hydra
from omegaconf import DictConfig, OmegaConf
import importlib


logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None, config_path="config/benchmark_gpu", config_name="default"
)
def main(cfg: DictConfig):
    OUTPUT_DIR = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    n_iter = cfg.n_iter  # default 10
    n_warmup = cfg.n_warmup  # default 10
    amp_dtype = cfg.amp_dtype  # default None
    compile_run_path = Path(cfg.compile_run_path)
    compile_run_cfg = OmegaConf.load(compile_run_path / ".hydra" / "config.yaml")
    print(OmegaConf.to_yaml(compile_run_cfg))

    # Setting variables
    for var, val in compile_run_cfg.env.items():
        logging.info(f"Setting {var} to {val}")
        module_name, attr_name = var.rsplit(".", 1)
        module = importlib.import_module(module_name)
        setattr(module, attr_name, val)

    height, width = compile_run_cfg.image.height, compile_run_cfg.image.width

    base_model = load_model(
        config_file=compile_run_cfg.model.config,
        ckpt_path=compile_run_cfg.model.ckpt_path,
        opts=compile_run_cfg.model.opts,
    )

    _, inputs = load_input_fixed(height=height, width=width, device="cuda")
    model = TracingAdapter(
        base_model, inputs=inputs, allow_non_tensor=False, specialize_non_tensor=True
    )

    inputs = model.flattened_inputs
    print(inputs[0].shape)

    if cfg.load_ts:
        del base_model, model
        model_path = compile_run_path / "model.ts"
        model = torch.jit.load(model_path)

    torch.cuda.memory._record_memory_history()

    model.eval()
    model.cuda()

    ctx = contextlib.nullcontext
    if amp_dtype is not None:
        if amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif amp_dtype == "bf16":
            amp_dtype = torch.bfloat16
        ctx = partial(torch.amp.autocast, "cuda", dtype=amp_dtype)

    with torch.no_grad(), ctx():
        logging.info("warmup")
        for _ in range(n_warmup):
            _ = model(*inputs)

        torch.cuda.reset_peak_memory_stats()
        logging.info("measuring time")
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = model(*inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            times.append(inference_time * 1e3)

        avg = mean(times)
        std = stdev(times)
        logging.info(f"Average inference time: {avg:.4f} ± {std:.4f}")

        # get max memory usage
        max_memory = torch.cuda.memory.max_memory_allocated()
        torch.cuda.memory._dump_snapshot(OUTPUT_DIR / "mem.pickle")
        logging.info(f"Max memory usage: {max_memory / 1e6:.4f} MB")


if __name__ == "__main__":
    main()
