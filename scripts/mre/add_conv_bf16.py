import tensorrt as trt
from torch_tensorrt.fx.types import TRTTensor
import numpy as np
import ml_dtypes

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network()
input = network.add_input(name="input", dtype=trt.bfloat16, shape=(1, 3, 512, 512))
weight = np.random.randn(3, 3, 3, 3).astype(ml_dtypes.bfloat16)
bias = np.random.randn(3).astype(ml_dtypes.bfloat16)
conv_layer = network.add_convolution_nd(
    input=input,
    num_output_maps=weight.shape[0],
    kernel_shape=weight.shape[2:],
    # kernel=trt.Weights() if isinstance(weight, TRTTensor) else weight,
    # bias=trt.Weights() if isinstance(bias, TRTTensor) else bias,
    kernel=weight,
    bias=bias,
)

config = builder.create_builder_config()