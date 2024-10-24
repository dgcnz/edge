import torch_tensorrt
import torch
import modelopt.torch.quantization as mtq
from src.utils.quantization import load_input_fixed, ModelWrapper, load_model, unflatten_repr, plot_predictions
# from torchao.utils import unwrap_tensor_subclass
from modelopt.torch.quantization.utils import export_torch_mode

img, example_kwargs = load_input_fixed(height=512, width=512)
model = ModelWrapper(
    net=load_model().cuda(),
    height=example_kwargs["heights"][0],
    width=example_kwargs["widths"][0],
)
model.eval().cuda()
inputs = (example_kwargs["images"].cuda(),)

# The quantization algorithm requires calibration data. Below we show a rough example of how to
# set up a calibration data loader with the desired calib_size
# data_loader = get_dataloader(num_samples=calib_size)


# Define the forward_loop function with the model as input. The data loader should be wrapped
# inside the function.
def forward_loop(model):
    for _ in range(10):
        model(*inputs)

_dtype = "int8"
if _dtype == "fp8":
    dtype = torch.float8_e4m3fn
    cfg = mtq.FP8_DEFAULT_CFG
    enabled_precisions = {dtype}
elif _dtype == "int8":
    dtype = torch.int8
    cfg = mtq.INT8_DEFAULT_CFG
    enabled_precisions = {dtype}
else:
    raise ValueError(f"Unsupported dtype: {_dtype}")
# Quantize the model and perform calibration (PTQ)
model = mtq.quantize(model, cfg, forward_loop)

out = model(*inputs)
print(out)

mtq.print_quant_summary(model)
with torch.no_grad(), export_torch_mode():
    with open(f"{_dtype}.onnx", "wb") as f:
        torch.onnx.export(model, inputs, f)

#with torch.no_grad(), export_torch_mode():
#    from torch.export._trace import _export
#
#    exported_program = _export(
#        model,
#        args=inputs,
#    )
#    with open("a.txt", "w") as f:
#        f.write(str(exported_program))
#    trt_gm = torch_tensorrt.dynamo.compile(
#        exported_program,
#        inputs,
#        reuse_cached_engines=False,
#        cache_built_engines=False,
#        enable_experimental_decompositions=True,
#        truncate_double=True,
#        use_fast_partitioner=True,
#        require_full_compilation=True,
#        # min_block_size=1,
#        enabled_precisions=enabled_precisions,
#        debug=True,
#    )
#    print(trt_gm)
#    outputs =  trt_gm(*inputs)
#    torch_tensorrt.save(trt_gm, f"{_dtype}.ts", output_format="torchscript", inputs=inputs)
#    outputs = unflatten_repr(outputs)
#    plot_predictions(outputs, img, output_file=f"{_dtype}.png")
#

# exported_program = torch.export.export(
#     model,
#     args=inputs,
# )
# enabled_precisions = {torch.float32, torch.bfloat16, torch.float16, torch.int8}
# trt_gm = torch_tensorrt.dynamo.compile(
#     exported_program,
#     inputs,
#     reuse_cached_engines=False,
#     cache_built_engines=False,
#     enable_experimental_decompositions=True,
#     truncate_double=True,
#     use_fast_partitioner=True,
#     require_full_compilation=True,
#     enabled_precisions=enabled_precisions,
# )
