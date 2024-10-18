import torch
# import torch_tensorrt
# import modelopt.torch.quantization as mtq
from src.utils.quantization import load_input_fixed, ModelWrapper, load_model
from torchao.utils import unwrap_tensor_subclass
import pickle
import torchao

img, example_kwargs = load_input_fixed(height=512, width=512)
model = ModelWrapper(
    net=load_model().cuda(),
    height=example_kwargs["heights"][0],
    width=example_kwargs["widths"][0],
)
model.eval().cuda()
inputs = (example_kwargs["images"].cuda(),)