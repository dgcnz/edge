import torchao
import logging
from torchao.quantization.quant_api import (
    quantize_,
    int8_dynamic_activation_int8_weight,
    int4_weight_only,
    int8_weight_only,
    unwrap_tensor_subclass,
)
import detectron2.structures
from src.utils.quantization import (
    load_input,
    load_model,
    register_DINO_output_types,
    load_input_fixed,
    flatten_detectron2_instances,
    unflatten_detectron2_instances,
    unflatten_detectron2_boxes,
    flatten_detectron2_boxes,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import detectron2
import torch_tensorrt
import torch
from copy import copy
import matplotlib.pyplot as plt
import torch
from torch_tensorrt.dynamo._exporter import inline_torch_modules


class ModelWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, height: int, width: int):
        super().__init__()
        self.net = net
        self.height = height
        self.width = width

    def forward(self, images: torch.Tensor):
        res = self.net(
            **{"images": images, "heights": [self.height], "widths": [self.width]}
        )[0]
        # res["instances"]._image_size = torch.tensor(res["instances"]._image_size)
        return flatten_repr(res)


def flatten_repr(obj):
    obj["instances"] = flatten_detectron2_instances(obj["instances"])[0]
    obj["instances"][1]["pred_boxes"] = flatten_detectron2_boxes(
        obj["instances"][1]["pred_boxes"]
    )[0][0]
    obj = obj["instances"][1]
    return (obj["pred_boxes"], obj["scores"], obj["pred_classes"])


def unflatten_repr(obj):
    obj = dict(pred_boxes=obj[0], scores=obj[1], pred_classes=obj[2])
    obj = dict(instances=[(512, 512), obj])
    obj["instances"][1]["pred_boxes"] = unflatten_detectron2_boxes(
        [obj["instances"][1]["pred_boxes"]], None
    )
    obj["instances"] = unflatten_detectron2_instances(obj["instances"], None)
    return obj


register_DINO_output_types()
img, example_kwargs = load_input_fixed(height=512, width=512)
model = load_model(device="cuda").cuda()
model = (
    ModelWrapper(
        net=model,
        height=example_kwargs["heights"][0],
        width=example_kwargs["widths"][0],
    )
    .eval()
    .cuda()
)
inputs = (example_kwargs["images"].cuda(),)
inputs = (torch.randn(1, 3, 512, 512).cuda(),)
# print(model(*inputs))
# quantize_(model, int8_dynamic_activation_int8_weight())
# print(model(*inputs))
# model = unwrap_tensor_subclass(model)
# print(model(*inputs))

logging.info("warmup")
for _ in range(5):
    _ = model(example_kwargs["images"].cuda())
# Measure inference time with GPU synchronization
import time
logging.info("warmup")
times = []
for _ in range(5):
    torch.cuda.synchronize()
    start_time = time.time()
    _ = model(example_kwargs["images"].cuda())
    torch.cuda.synchronize()
    end_time = time.time()
    inference_time = end_time - start_time
    times.append(inference_time)
    print(inference_time)

print(times)