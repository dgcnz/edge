import torch_tensorrt
from typing import Optional
import logging
from contextlib import nullcontext
from src.utils.quantization import (
    load_model,
    plot_predictions,
    load_input_fixed,
    ModelWrapper,
    unflatten_repr,
)
import torch
import argparse
from functools import partial

logging.basicConfig(level=logging.INFO)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
        help="AMP dtype. If fp32, no AMP is used.",
    )
    parser.add_argument(
        "--trt_precisions",
        nargs="+",
        help="List of possible precisions for TensorRT",
        required=True,
        # choices=[
        #     ["fp32"],
        #     ["bf16"],
        #     ["fp16"],
        #     ["fp32", "bf16"],
        #     ["fp32", "fp16"],
        #     ["bf16", "fp16"],
        #     ["fp32", "fp16", "bf16"],
        # ],
    )

    return parser


def to_dtype(precision: str):
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def compile(
    model: torch.nn.Module, inputs: tuple, amp_dtype: str = "fp32", trt_precisions: list[str] = ["fp32"]
):
    """
    Compile the model with the given AMP dtype and enabled TRT precisions.
    :param model: Model to compile
    :param inputs: Example inputs
    :param amp_dtype: AMP dtype, if fp32, no AMP is used
    :param trt_precisions: List of precisions to enable in TensorRT
    """

    amp_dtype = to_dtype(amp_dtype)
    amp_autocast = nullcontext
    if amp_dtype != torch.float32:
        amp_autocast = partial(torch.amp.autocast, "cuda", dtype=amp_dtype)

    enabled_precisions = {to_dtype(p) for p in trt_precisions}

    logging.info(f"Compiling model with {amp_dtype} and {trt_precisions}")

    with torch.no_grad(), amp_autocast():
        logging.info("Warmup")
        for _ in range(2):
            model(*inputs)

        logging.info("Exporting model")
        exported_program = torch.export.export(
            model,
            args=inputs,
        )
        logging.info("Compiling model")
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs,
            reuse_cached_engines=False,
            cache_built_engines=False,
            enable_experimental_decompositions=True,
            truncate_double=True,
            use_fast_partitioner=True,
            require_full_compilation=True,
            enabled_precisions=enabled_precisions,
        )
        logging.info("Compiled model")
        return trt_gm


def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    # check that amp_dtype is in enabled_precisions
    if args.amp_dtype not in args.trt_precisions:
        raise ValueError(
            f"amp_dtype {args.amp_dtype} is not in trt_precisions {args.trt_precisions}"
        )
    args.trt_precisions.sort() # to ensure consistency in output file name

    logging.info("Loading model and example input")
    img, example_kwargs = load_input_fixed(height=args.height, width=args.width)
    model = ModelWrapper(
        net=load_model().cuda(),
        height=example_kwargs["heights"][0],
        width=example_kwargs["widths"][0],
    )
    model.eval().cuda()
    inputs = (example_kwargs["images"].cuda(),)
    trt_gm = compile(
        model, inputs, amp_dtype=args.amp_dtype, trt_precisions=args.trt_precisions
    )

    logging.info("Executing compiled model")
    outputs = trt_gm(*inputs)
    outputs = unflatten_repr(outputs)
    logging.info(f"Predictions\n{outputs}")

    logging.info("Plotting predictions")
    enabled_precisions = "-".join(args.trt_precisions)  
    output_name = f"{args.amp_dtype}_{enabled_precisions}_{args.height}_{args.width}"
    plot_predictions(outputs, img, output_file=f"{output_name}.png")
    try:
        logging.info("Saving PT2E")
        torch_tensorrt.save(trt_gm, f"artifacts/{output_name}.ep", inputs=inputs)
    except Exception as e:
        logging.error("Failed to save PT2E", exc_info=True)

    try:
        logging.info("Saving TorchScript")
        torch_tensorrt.save(
            trt_gm,
            f"artifacts/{output_name}.ts",
            output_format="torchscript",
            inputs=inputs,
        )
    except Exception as e:
        logging.error("Failed to save TorchScript", exc_info=True)


if __name__ == "__main__":
    main()
