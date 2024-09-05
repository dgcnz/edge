import timm
from torch.export import export
import torch
from detrex.modeling.backbone import TimmBackbone
from functools import partial
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
def export_and_quantize(model: VisionTransformer, example_input: tuple[torch.Tensor]):
    exported_program_strict: torch.export.ExportedProgram = export(
        model,
        args=example_input,
        strict=True,
    )
    m = exported_program_strict.module()
    m.model_name = model.model_name
    m = prepare_pt2e(m, quantizer)
    m = convert_pt2e(m)

dinov2: VisionTransformer = timm.create_model(
    model_name="vit_base_patch14_dinov2.lvd142m",
    pretrained=False,
)
dinov2.model_name = "dinov2"
vit: VisionTransformer = timm.create_model(
    model_name="vit_base_patch16_224",
    pretrained=False,
)
vit.model_name = "vit"
assert isinstance(dinov2, VisionTransformer), "model is not VisionTransformer"
assert isinstance(vit, VisionTransformer), "model is not VisionTransformer"
input_dinov2 = (torch.randn(1, 3, 518, 518),)
input_vit = (torch.randn(1, 3, 224, 224),)

export_and_quantize(vit, input_vit)
print("SUCCESS")
export_and_quantize(dinov2, input_dinov2)
print("SUCCESS")
