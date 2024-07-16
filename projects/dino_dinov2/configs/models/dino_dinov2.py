from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detrex.modeling.backbone import TimmBackbone, SimpleFeaturePyramid


from .dino_vitdet import model


# Base

model.backbone = L(SimpleFeaturePyramid)(
    net=L(TimmBackbone)(
        model_name="vit_base_patch14_dinov2.lvd142m",
        features_only=True,
        pretrained=True,
        in_channels=3,
        out_indices=(-1,),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        patch_size=16,
        freeze=True,
    ),
    in_feature="p-1",
    out_channels=256,
    scale_factors=(2.0, 1.0, 0.5),  # (4.0, 2.0, 1.0, 0.5) in ViTDet
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=518,
)

# modify neck config
model.neck.input_shapes = {
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}
model.neck.in_features = ["p3", "p4", "p5", "p6"]
model.neck.num_outs = 4
model.transformer.num_feature_levels = 4