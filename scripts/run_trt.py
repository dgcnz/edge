import torch
torch.cuda.memory._record_memory_history()
from src.utils.quantization import load_input_fixed,  unflatten_repr, filter_predictions_with_confidence
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from demo import VisualizationDemo
from detectron2.data.detection_utils import read_image
from copy import copy
import torchvision.transforms as T
import torch_tensorrt
from torch.profiler import profile, record_function, ProfilerActivity
import logging
# import nvidia_dlprof_pytorch_nvtx
# nvidia_dlprof_pytorch_nvtx.init()




logging.basicConfig(level=logging.INFO)

logging.info("Loading model")
# model = torch.jit.load("trt_f16.ts")
# model = torch.export.load("trt_f16.ep").module()
model = torch.jit.load("trt.ts")
# model = torch.export.load("trt.ep").module()
logging.info("Loaded model")

image_path = "artifacts/idea_raw.jpg"
# image_path = "artifacts/ams.jpg"
# image_path = "artifacts/white.jpg"
img, example_kwargs = load_input_fixed(image_path, height=512, width=512)




# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    #with record_function("model_inference"):
# logging.info("Inference")
# #with torch.autograd.profiler.emit_nvtx():

# for i in range(10):
#     with torch.no_grad():
#         logging.info(f"Forward pass {i}")
#         outputs = model(example_kwargs["images"].cuda())

# 
# 
# # print(torch.cuda.max_memory_allocated())
# # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
# 
with torch.no_grad():
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

    print(times)