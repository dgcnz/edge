# https://github.com/pytorch/TensorRT/issues/3226
# import modelopt.torch.quantization as mtq
import torch
import torch_tensorrt
import torch.nn.functional as F
# import torchao

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, _, h, w = x.shape
        z = F.interpolate(x, [h*2, w*2])
        return z

model = Model().cuda()
inputs = (torch.randn((2, 4, 8, 8)).cuda(),)

with torch.no_grad():
    ep = torch.export.export(
        model,
        args=inputs,
        strict=True
    )
    with torch_tensorrt.logging.debug():
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            inputs,
            reuse_cached_engines=False,
            cache_built_engines=False,
            require_full_compilation=True,
            min_block_size=1,
        )  