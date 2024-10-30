import logging
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import hydra
import torch
import torch_tensorrt
from omegaconf import DictConfig, OmegaConf

import importlib
import detrex
from src.utils import TracingAdapter, load_input_fixed, load_model, plot_predictions

logging.basicConfig(level=logging.INFO)
# torch._subclasses.fake_tensor.CONSTANT_NUMEL_LIMIT = 2000
# detrex.layers.multi_scale_deform_attn._ENABLE_CUDA_MSDA = False


def to_dtype(precision: str):
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def compile(
    model: torch.nn.Module,
    inputs: tuple,
    trt_cfg: DictConfig,
    amp_dtype: str = "fp32",
):
    """
    Compile the model with the given AMP dtype and enabled TRT precisions.
    :param model: Model to compile
    :param inputs: Example inputs
    :param trt_cfg: TRT compilation configuration
    :param amp_dtype: AMP dtype, if fp32, no AMP is used
    """

    amp_dtype = to_dtype(amp_dtype)
    amp_autocast = nullcontext
    if amp_dtype != torch.float32:
        amp_autocast = partial(torch.amp.autocast, "cuda", dtype=amp_dtype)

    assert trt_cfg.enabled_precisions, "enabled_precisions must not be empty"
    trt_cfg = OmegaConf.to_container(trt_cfg, resolve=True)
    trt_cfg["enabled_precisions"] = list(
        set(to_dtype(p) for p in trt_cfg["enabled_precisions"])
    )

    logging.info(
        f"Compiling model with {amp_dtype} and {trt_cfg['enabled_precisions']}"
    )

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
        trt_kwargs = {
            "reuse_cached_engines": False,
            "cache_built_engines": False,
            "enable_experimental_decompositions": True,
            "truncate_double": True,
            "use_fast_partitioner": True,
            "require_full_compilation": True,
            "debug": True,
        }
        # override trt_kwargs with trt_cfg
        trt_kwargs.update(trt_cfg)
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs,
            **trt_kwargs,
        )
        logging.info("Compiled model")
        return trt_gm


@hydra.main(version_base=None, config_path="config/export_tensorrt", config_name="dinov2")
def main(cfg: DictConfig):
    OUTPUT_DIR = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))

    # Setting variables
    for var, val in cfg.env.items():
        logging.info(f"Setting {var} to {val}")
        module_name, attr_name = var.rsplit(".", 1)
        module = importlib.import_module(module_name)
        setattr(module, attr_name, val)

    # check that amp_dtype is in enabled_precisions
    if cfg.amp_dtype not in cfg.trt.enabled_precisions:
        raise ValueError(
            f"amp_dtype {cfg.amp_dtype} is not in cfg.trt.enabled_precisions {cfg.trt.enabled_precisions}"
        )
    cfg.trt.enabled_precisions.sort()  # to ensure consistency in output file name
    # save cfg to yaml file using OmegaConf

    logging.info("Loading model and example input")
    img, raw_inputs = load_input_fixed(
        image_path=cfg.image.path,
        height=cfg.image.height,
        width=cfg.image.width,
        device="cuda",
    )
    model = load_model(
        config_file=cfg.model.config,
        ckpt_path=cfg.model.ckpt_path,
        opts=cfg.model.opts,
    ).cuda()
    model = TracingAdapter(
        model, inputs=raw_inputs, allow_non_tensor=False, specialize_non_tensor=True
    )
    inputs = model.flattened_inputs
    model.eval().cuda()
    # This forward call is important, it ensures the model works before compilation
    model(*inputs)
    try:
        trt_gm = compile(model, inputs, amp_dtype=cfg.amp_dtype, trt_cfg=cfg.trt)
    except Exception as e:
        logging.error("Failed to compile model", exc_info=True)
        return

    logging.info("Executing compiled model")
    outputs = trt_gm(*inputs)
    outputs = model.outputs_schema(outputs)[0]
    logging.info(f"Predictions\n{outputs}")

    logging.info("Plotting predictions")
    plot_predictions(outputs, img, output_file=str(OUTPUT_DIR / "predictions.png"))
    try:
        logging.info("Saving PT2E")
        torch_tensorrt.save(trt_gm, str(OUTPUT_DIR / "model.pt2"), inputs=inputs)
    except Exception as e:
        logging.error("Failed to save PT2E", exc_info=True)
        logging.info("Saved PT2E")

    try:
        logging.info("Saving TorchScript")
        torch_tensorrt.save(
            trt_gm,
            str(OUTPUT_DIR / f"model.ts"),
            output_format="torchscript",
            inputs=inputs,
        )
        logging.info("Saved TorchScript")
    except Exception as e:
        logging.error("Failed to save TorchScript", exc_info=True)


if __name__ == "__main__":
    main()
