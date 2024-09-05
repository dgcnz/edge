import timm
import torch
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

model = timm.create_model("vit_base_patch14_dinov2.lvd142m")
example_input = (torch.randn(1, 3, 518, 518),)

quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
exported_program_strict = torch.export.export(
    model,
    args=example_input,
    strict=True,
)
m = exported_program_strict.module()
m = prepare_pt2e(m, quantizer)
print("SUCCESS")