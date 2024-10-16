import torch
import time
from typing import Optional
from functools import partial
import contextlib
from src.utils.quantization import (
    load_input_fixed,
    unflatten_repr,
    plot_predictions,
    ModelWrapper,
)
from src.utils.quantization import load_model as _load_model
from statistics import stdev, mean
import torch_tensorrt
import logging
import argparse
from pathlib import Path


def setup_parser():
    DEFAULT_IMG = Path("artifacts/idea_raw.jpg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMG)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--amp_dtype", type=str, default=None, choices=["fp16", "bf16", None])
    return parser


logging.basicConfig(level=logging.INFO)


def load_model(model_path: Path):
    if model_path.suffix == ".ts":
        *_, height, width = model_path.stem.split("_")
        model = torch.jit.load(model_path)
    elif model_path.suffix == ".ep":
        *_, height, width = model_path.stem.split("_")
        model = torch.export.load(model_path).module()
    elif model_path.suffix == ".pth":
        height, width = 512, 512
        model = ModelWrapper(
            net=_load_model().cuda(),
            height=height,
            width=width,
        )
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

    return model, int(height), int(width)


def benchmark(
    model_path: Path,
    image_path: Path,
    n_warmup: int,
    n_iter: int,
    output_path: Optional[Path],
    amp_dtype: Optional[str] = None,
):
    # track cuda memory history
    torch.cuda.memory._record_memory_history()
    model, height, width = load_model(model_path)
    model.eval()
    model.cuda()
    logging.info("Loaded model")
    img, example_kwargs = load_input_fixed(str(image_path), height, width)
    input = (example_kwargs["images"].cuda(),)

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
            _ = model(*input)

        torch.cuda.reset_peak_memory_stats()
        logging.info("measuring time")
        times = []
        for _ in range(n_iter):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = model(*input)
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            times.append(inference_time * 1e3)

        avg = mean(times)
        std = stdev(times)
        logging.info(f"Average inference time: {avg:.4f} ± {std:.4f}")

        # get max memory usage
        max_memory = torch.cuda.memory.max_memory_allocated()
        torch.cuda.memory._dump_snapshot(f"artifacts/{model_path.stem}_mem.pickle")
        logging.info(f"Max memory usage: {max_memory / 1e6:.4f} MB")

    if output_path is not None:
        outputs = model(*input)
        outputs = unflatten_repr(outputs)
        plot_predictions(outputs, img, output_file=output_path)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    logging.info("Loading model")
    model_path = args.model
    benchmark(
        model_path, args.image, args.n_warmup, args.n_iter, args.output, args.amp_dtype
    )


if __name__ == "__main__":
    main()
