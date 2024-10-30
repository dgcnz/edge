from copy import copy

import detectron2.data.transforms as T
import matplotlib.pyplot as plt
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer
from omegaconf import OmegaConf


def filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "instances" in predictions:
        preds = predictions["instances"]
        keep_idxs = preds.scores > confidence_threshold
        predictions = copy(predictions)  # don't modify the original
        predictions["instances"] = preds[keep_idxs]
    return predictions


def load_input_fixed(
    image_path: str = "artifacts/idea_raw.jpg",
    height: int = 800,
    width: int = 1200,
    input_format: str = "RGB",
    device: str = "cuda",
):
    img = read_image(image_path, format="BGR")
    res = T.Resize((height, width))
    original_img = res.get_transform(img).apply_image(img)
    img = original_img.copy()
    with torch.no_grad():
        if input_format == "RGB":
            img = img[:, :, ::-1]
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return original_img, (
            [
                {
                    "image": img.to(device),
                    "height": height,
                    "width": width,
                }
            ],
        )


def get_opts(config_file: str, img_size: tuple[int, int]) -> list[str]:
    if config_file == "projects/dino_dinov2/configs/models/dino_dinov2.py":
        """
        # EXPORT_RULE:
        # - disable dynamic image sizes
        # - fix image size to target device
        """
        return [
            f"model.backbone.net.img_size={list(img_size)}",
            f"model.backbone.net.dynamic_img_size=False",
            f"model.backbone.net.dynamic_img_pad=False",
        ]
    elif (
        config_file
        == "detrex/detectron2/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_b_100ep.py"
    ):
        # doesn't allow dynamic image sizes
        assert img_size == (
            1024,
            1024,
        ), "pre-trained detetron2.backbone.ViT doesn't support interpolation"
    else:
        return []


def load_model(
    config_file: str = "projects/dino_dinov2/configs/models/dino_dinov2.py",
    ckpt_path: str = "artifacts/model_final.pth",
    opts: list[str] = []
) -> torch.nn.Module:
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opts)
    model: torch.nn.Module = instantiate(OmegaConf.to_object(cfg.model)).eval()
    if ckpt_path:
        print(f"Loading checkpoint {ckpt_path}")
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(ckpt_path)
    return model


def plot_predictions(outputs, img, display: bool = False, output_file: str = "res.png"):
    pred = filter_predictions_with_confidence(outputs, confidence_threshold=0.5)
    v = Visualizer(img, MetadataCatalog.get("coco_2017_val"))
    v = v.draw_instance_predictions(pred["instances"].to("cpu"))
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(v.get_image()[:, :, ::-1])
    ax.axis("off")
    if display:
        plt.show()
    plt.savefig(output_file)
    return fig, ax, v
