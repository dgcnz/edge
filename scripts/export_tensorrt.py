import torch_tensorrt
import logging
from src.utils.quantization import (
    load_model,
    register_DINO_output_types,
    load_input_fixed,
    ModelWrapper,
    unflatten_repr,
    filter_predictions_with_confidence,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
import matplotlib.pyplot as plt
import torch

logging.basicConfig(level=logging.INFO)

register_DINO_output_types()
img, example_kwargs = load_input_fixed(height=512, width=512)
model = ModelWrapper(
    net=load_model().cuda(),
    height=example_kwargs["heights"][0],
    width=example_kwargs["widths"][0],
)
model.eval().cuda()
inputs = (example_kwargs["images"].cuda(),)

with torch.no_grad():# , torch.amp.autocast("cuda", dtype=torch.float16):
    for _ in range(2):
        model(*inputs)

    exported_program = torch.export.export(
        model,
        args=inputs,
    )

    trt_gm = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs,
        reuse_cached_engines=False,
        cache_built_engines=False,
        enable_experimental_decompositions=True,
        truncate_double=True,
        use_fast_partitioner=True,
        require_full_compilation=True,
        # optimization_level=5,
        enabled_precisions = {torch.float32, torch.float16}
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
