from src.utils.quantization import load_input, load_model, register_DINO_output_types
import torch_tensorrt
import torch


class ModelWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, height: int, width: int):
        super().__init__()
        self.net = net
        self.height = height
        self.width = width

    def forward(self, images: torch.Tensor):
        return self.net(
            **{"images": images, "heights": [self.height], "widths": [self.width]}
        )


register_DINO_output_types()
example_kwargs = load_input()
model = load_model(device="cuda").cuda()
model = ModelWrapper(
    net=model, height=example_kwargs["heights"][0], width=example_kwargs["widths"][0]
).eval().cuda()
inputs = (example_kwargs["images"].cuda(),)
model(*inputs)
exported_program = torch.export.export(
    model,
    args=inputs,
)
with torch_tensorrt.logging.debug():
    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs,
        # reuse_cached_engines=False,
        # cache_built_engines=True,
        enable_experimental_decompositions=True,
        truncate_double=True,
        make_refitable=True,
    )  # Output is a torch.fx.GraphModule
trt_gm(*inputs)
