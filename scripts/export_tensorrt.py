import torch_tensorrt
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
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    return parser


def compile(model, inputs: tuple, half: bool = False, bf16: bool = False):
    fp16_autocast = partial(torch.amp.autocast, "cuda", dtype=torch.float16)
    ctx = fp16_autocast if half else nullcontext

    enabled_precisions = {torch.float32, torch.float16} if half else {torch.float32}
    if bf16:
        # bf16 doesn't affect the model's precision
        enabled_precisions.add(torch.bfloat16)

    logging.info(f"Compiling model with {'fp16' if half else 'fp32'}")

    with torch.no_grad(), ctx():
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
        logging.info("Compiled model c: ")
        return trt_gm


def main():
    parser = setup_parser()
    args = parser.parse_args()

    logging.info("Loading model and example input")
    img, example_kwargs = load_input_fixed(height=args.height, width=args.width)
    model = ModelWrapper(
        net=load_model().cuda(),
        height=example_kwargs["heights"][0],
        width=example_kwargs["widths"][0],
    )
    model.eval().cuda()
    inputs = (example_kwargs["images"].cuda(),)

    trt_gm = compile(model, inputs, half=args.half, bf16=args.bf16)

    logging.info("Executing compiled model")
    outputs = trt_gm(*inputs)
    outputs = unflatten_repr(outputs)
    logging.info(f"Predictions\n{outputs}")

    logging.info("Plotting predictions")
    precision = "fp16" if args.half else "fp32"
    if args.bf16:
        precision = precision+"-bf16"
    output_name = f"{precision}_{args.height}_{args.width}"
    plot_predictions(outputs, img, output_file=f"{output_name}.png")

    logging.info("Saving TorchScript")
    torch_tensorrt.save(
        trt_gm,
        f"artifacts/{output_name}.ts",
        output_format="torchscript",
        inputs=inputs,
    )
    logging.info("Saving PT2E")
    torch_tensorrt.save(trt_gm, f"artifacts/{output_name}.ep", inputs=inputs)


if __name__ == "__main__":
    main()
