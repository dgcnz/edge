import torch.nn.functional as F
from typing import List, Tuple
import torch
import detectron2
import detectron2.structures

def batch_images(image_batch: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    if len(image_batch) == 1:
        return torch.stack(image_batch), [(image_batch[0].size(1),  image_batch[0].size(2))]

    image_shapes: List[Tuple[int, int]] = [(0, 0)] * len(image_batch)
    max_height, max_width = 0, 0
    for ix, img in enumerate(image_batch):
        image_shapes[ix] = (img.size(1), img.size(2))
        max_height = max(max_height, image_shapes[ix][0])
        max_width = max(max_width, image_shapes[ix][1])

    image_batch = [
        F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
        for img in image_batch
    ]
    return torch.stack(image_batch), image_shapes


class MRE(torch.nn.Module):
    def forward(self, images: List[torch.Tensor]):
        # images_tensor = batch_images(images)
        # tensors = detectron2.structures.ImageList(images_tensor, [(24, 24)]).tensor
        tensors = detectron2.structures.ImageList.from_tensors(images).tensor
        return 2 * tensors
    
class HotFix(torch.nn.Module):
    def forward(self, images: List[torch.Tensor]):
        images_tensor = batch_images(images)
        dims = [tuple(img.shape[1:]) for img in images]
        tensors = detectron2.structures.ImageList(images_tensor, dims).tensor
        return 2 * tensors
    

example_kwargs = {
    "images": [torch.randn(3, 24, 24)],
}
results = {}
models = {"HotFix": HotFix(), "MRE": MRE()}
for model_name, model in models.items():
    try:
        torch.export.export(model, (), kwargs=example_kwargs)
        results[model_name] = "SUCCESS"
    except Exception as e:
        results[model_name] = str(e)

print(results)