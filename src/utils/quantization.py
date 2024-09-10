from detectron2.data.detection_utils import read_image
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import torch
import logging
from pathlib import Path
import torch.fx
from omegaconf import OmegaConf
import detectron2.data.transforms as T
import typing
import torch.utils._pytree as pytree
import detectron2.structures.instances
import detectron2.export
import torch
from torch.export import export
import typing
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
import torchvision
import torchvision.transforms.functional

def load_input(
    image_path: str = "artifacts/idea_raw.jpg",
    min_size_test: int = 800,
    max_size_test: int = 1333,
    input_format: str = "RGB",
) -> dict[str, typing.Any]:
    img = read_image(image_path, format="BGR")
    aug = T.ResizeShortestEdge([min_size_test, min_size_test], max_size_test)
    original_image = img.copy()
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        if input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        return {
            "images": image.unsqueeze(0),
            "heights": [height],
            "widths": [width],
        }
    
def load_input_fixed(
    image_path: str = "artifacts/idea_raw.jpg",
    height: int = 800,
    width: int = 1200,
    input_format: str = "RGB",
):
    img = read_image(image_path, format="BGR")
    res = T.Resize((height, width))
    original_img = res.get_transform(img).apply_image(img)
    img = original_img.copy()
    with torch.no_grad(): 
        if input_format == "RGB":
            img = img[:, :, ::-1]
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return original_img, {
            "images": img.unsqueeze(0),
            "heights": [height],
            "widths": [width],
        }


def load_model(
    config_file: str = "projects/dino_dinov2/configs/COCO/dino_dinov2_b_12ep.py",
    ckpt_path: str = "artifacts/model_final.pth",
    device: str = "cpu",
) -> torch.nn.Module:
    opts = [
        f"model.device={device}",
        f"train.device={device}",
        f"train.init_checkpoint={ckpt_path}",
    ]
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    model: torch.nn.Module = instantiate(OmegaConf.to_object(cfg.model)).eval()
    if ckpt_path:
        print(f"Loading checkpoint {ckpt_path}")
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.train.init_checkpoint)
    return model

def unflatten_detectron2_boxes(values, _):
    boxes = object.__new__(detectron2.structures.boxes.Boxes)
    boxes.tensor = values[0]
    return boxes

def unflatten_detectron2_instances(values, _):
    instances = object.__new__(detectron2.structures.instances.Instances)
    instances._image_size = values[0]
    instances._fields = values[1]
    return instances

def register_DINO_output_types(): 
    pytree.register_pytree_node(
        detectron2.structures.boxes.Boxes,
        lambda x: ([x.tensor], None),
        unflatten_fn=unflatten_detectron2_boxes,
        serialized_type_name="detectron2.structures.boxes.Boxes",
    )

    pytree.register_pytree_node(
        detectron2.structures.instances.Instances,
        lambda x: ([x._image_size, x._fields], None),
        unflatten_fn=unflatten_detectron2_instances,
        serialized_type_name="detectron2.structures.instances.Instances",
    )


def export_dinov2(
    model: torch.nn.Module,
    example_kwargs: dict[str, typing.Any],
    cache: bool = False,
    exported_ckpt_path: typing.Optional[Path] = None,
) -> torch.export.ExportedProgram:
    try:
        register_DINO_output_types()
    except ValueError as e:
        logging.info(e)
    if cache and exported_ckpt_path is not None and exported_ckpt_path.exists():
        return torch.export.load(exported_ckpt_path)

    # torch._dynamo.config.verbose = True
    # torch._logging.set_logs(dynamo=logging.INFO)
    exported_program: torch.export.ExportedProgram = export(
        model,
        args=(),
        kwargs=example_kwargs,
        strict=True,
    )
    if cache:
        torch.export.save(exported_program, exported_ckpt_path)
    return exported_program


def static_quantize(
    model: torch.fx.GraphModule,
    example_kwargs: dict[str, typing.Any],
    cache: bool = False,
    ckpt_path: typing.Optional[Path] = None,
    # kind: "pt2e" | "eager" = "pt2e",
) -> torch.nn.Module:
    if cache and ckpt_path is not None and ckpt_path.exists():
        return load_quantized_model(ckpt_path)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    quantized_model = prepare_pt2e(model, quantizer)
    quantized_model = convert_pt2e(quantized_model)
    if cache:
        save_quantized_model(quantized_model, ckpt_path, example_kwargs)
    return quantized_model



def save_quantized_model(
    model: torch.nn.Module, ckpt_path: str, example_kwargs: dict[str, typing.Any]
):
    # 1. Export the model and Save ExportedProgram
    # capture the model to get an ExportedProgram
    quantized_ep = torch.export.export(
        model, args=(), kwargs=example_kwargs, strict=True
    )
    # use torch.export.save to save an ExportedProgram
    torch.export.save(quantized_ep, ckpt_path)


def load_quantized_model(ckpt_path: str):
    loaded_quantized_ep = torch.export.load(ckpt_path)
    loaded_quantized_model = loaded_quantized_ep.module()
    return loaded_quantized_model


def export_and_quantize_dinov2(
    example_kwargs: dict[str, typing.Any],
    exported_ckpt_path: typing.Optional[Path] = None,
    quantized_ckpt_path: typing.Optional[Path] = None,
    cache: bool = False,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    model = load_model(
        config_file="projects/dino_dinov2/configs/COCO/dino_dinov2_b_12ep.py",
        ckpt_path="artifacts/model_final.pth",
        device="cpu",
    ).eval()
    exported_program = export_dinov2(
        model, example_kwargs, cache=cache, exported_ckpt_path=exported_ckpt_path
    )
    exit(0)
    quantized_model = static_quantize(
        exported_program.module(),
        example_kwargs,
        cache=cache,
        ckpt_path=quantized_ckpt_path,
    )
    return model, exported_program, quantized_model
