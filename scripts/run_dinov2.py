import time
import logging
from src.utils.quantization import (
    load_model,
    register_DINO_output_types,
    load_input_fixed,
)
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch
from src.utils.quantization import (
    ModelWrapper,
    flatten_repr,
    unflatten_repr,
    filter_predictions_with_confidence,
)

register_DINO_output_types()
img, example_kwargs = load_input_fixed(height=512, width=512)
model = ModelWrapper(
    net=load_model().cuda(),
    height=example_kwargs["heights"][0],
    width=example_kwargs["widths"][0],
)
model.eval().cuda()

with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
    logging.info("warmup")
    for _ in range(5):
        _ = model(example_kwargs["images"].cuda())
    # Measure inference time with GPU synchronization
    logging.info("warmup")
    times = []

    for _ in range(5):
        torch.cuda.synchronize()
        start_time = time.time()
        out = model(example_kwargs["images"].cuda())
        torch.cuda.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        times.append(inference_time)

print(times)

outputs = unflatten_repr(out)
pred = filter_predictions_with_confidence(outputs, confidence_threshold=0.5)
v = Visualizer(img, MetadataCatalog.get("coco_2017_val"))
v = v.draw_instance_predictions(pred["instances"].to("cpu"))

# Display the results
plt.figure(figsize=(14, 10))
plt.imshow(v.get_image()[:, :, ::-1])
plt.axis("off")
plt.savefig("res.png")
