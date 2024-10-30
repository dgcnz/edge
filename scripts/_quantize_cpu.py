import torch
import torchao
import detectron2.structures
from src.utils._quantization import (
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
import torch
from copy import copy
import matplotlib.pyplot as plt
import torch


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
model = load_model(device="cpu").cpu()
model = (
    ModelWrapper(
        net=model,
        height=example_kwargs["heights"][0],
        width=example_kwargs["widths"][0],
    )
    .eval()
    .cpu()
)
with torch.no_grad():
    inputs = (example_kwargs["images"].cpu(),)
    # model(*inputs)
    model = torchao.autoquant(torch.compile(model, mode='max-autotune', fullgraph=True))
    for _ in range(2):
        model(*inputs)
    # exported_program = torch.export.export(
    #     model,
    #     args=inputs,
    # )
    # model = exported_program.module()

# model = torchao.autoquant(exported_program.module())



def filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "instances" in predictions:
        preds = predictions["instances"]
        keep_idxs = preds.scores > confidence_threshold
        predictions = copy(predictions)  # don't modify the original
        predictions["instances"] = preds[keep_idxs]
    return predictions

with torch.inference_mode(), torch.no_grad():
    outputs = model(*inputs)
    print(outputs)
    outputs = unflatten_repr(outputs)
    print(outputs)
    # outputs = outputs[0]
    pred = filter_predictions_with_confidence(outputs, confidence_threshold=0.5)
    v = Visualizer(img, MetadataCatalog.get("coco_2017_val"))
    v = v.draw_instance_predictions(pred["instances"].to("cpu"))

    # Display the results
    plt.figure(figsize=(14, 10))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.axis("off")
    plt.savefig("res.png")
    print("SAVE EP")
    # trt_gm = inline_torch_modules(trt_gm)
