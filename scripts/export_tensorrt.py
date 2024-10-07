import torchao
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
        net=model.half(),
        height=example_kwargs["heights"][0],
        width=example_kwargs["widths"][0],
    )
    .half()
    .eval()
    .cuda()
)
inputs = (example_kwargs["images"].cuda().half(),)
# print(model(*inputs))
# quantize_(model, int8_dynamic_activation_int8_weight())
# print(model(*inputs))
# model = unwrap_tensor_subclass(model)
# print(model(*inputs))
print("exporting program")
# with torch.inference_mode(), torch.no_grad():
exported_program = torch.export.export(
    model,
    args=inputs,
    strict=True,
)

# exported_program = exported_program.run_decompositions()

print("exported program")


def filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "instances" in predictions:
        preds = predictions["instances"]
        keep_idxs = preds.scores > confidence_threshold
        predictions = copy(predictions)  # don't modify the original
        predictions["instances"] = preds[keep_idxs]
    return predictions


with torch.no_grad():
    with torch_tensorrt.logging.debug():
        #trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
        trt_gm = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs,
            reuse_cached_engines=False,
            cache_built_engines=False,
            enable_experimental_decompositions=True,
            truncate_double=True,
            use_fast_partitioner=True,
            optimization_level=5,
            require_full_compilation=True,
            enabled_precisions = {torch.half}
            # make_refitable=True,
        )  # Output is a torch.fx.GraphModule
        print("OUTPUT OF COMPILED MODEL")
        outputs = trt_gm(*inputs)
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

        print("BYE")
        print("SAVE TS")
        torch_tensorrt.save(
            trt_gm, "trt.ts", output_format="torchscript", inputs=inputs
        )
        print("SAVE EP")
        # trt_gm = inline_torch_modules(trt_gm)
        torch_tensorrt.save(trt_gm, "trt.ep", inputs=inputs)
