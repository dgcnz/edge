# State of the Art

| Encoder              | Model                                                                                                       | Version | COCO  <br>(minival)                              | FPS                      | Device         | Code + <br>Weights |
| -------------------- | ----------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------ | ------------------------ | -------------- | ------------------ |
| **CNN** / **Hybrid** | YOLOv10                                                                                                     | L       | 53.4                                             | 137                      | T4[^1] | ✅                  |
|                      |                                                                                                             | X       | 54.4                                             | 93                       | T4[^1] | ✅                  |
|                      | YOLOv8                                                                                                      | L       | 52.9                                             | 81                       | T4[^1] | ✅                  |
|                      |                                                                                                             | X       | 53.9                                             | 59                       | T4[^1] | ✅                  |
|                      | RT-DETR {cite}`rtdetr`| R50     | 53.1                                             | 108                      | T4[^1] | ✅                  |
|                      |                                                                                                             | R101    | 54.3                                             | 73                       | T4[^1] | ✅                  |
| **ViT (ts mod)**     | MViTv2 {cite}`mvitv2` | L       | 55.7                                             | 5                        | A100           | ✅                  |
|                      |                                                                                                             | H       | 55.8                                             | 3                        | A100           | ✅                  |
|                      | Co-DETR {cite}`codetr`                                           | Swin-L  | **65.9**                                         | -                        | -              | ✅                  |
| **ViT (plain)**      | ViTDet {cite}`vitdet`     | ViT-B   | 54                                               | 11                       | A100           | ✅<br>              |
|                      |                                                                                                             | ViT-L   | 57.6<br>59.6<sup>2<sup/><br>-                    | 7                        | A100           | ✅<br>              |
|                      |                                                                                                             | ViT-H   | 58.7<br>**60.4**[^2]<br>-                | 5                        | A100           | ✅<br>              |
|                      | SimPLR {cite}`simplr` | ViT-L   | 58.7                                             | 9                        | A100           | ❌      |
|                      |                                                                                                             | ViT-H   | 59.8                                             | 7                        | A100           | ❌  |
| **ViT (mod)**        | EVA-02 {cite}`eva02` | TrV-L   | 59.2<br>62.3[^2]<br>**64.1**[^2] | ~1.4x faster<br>than ViT | A100           | ✅<br>              |

[^1]: With TensorRT FP16.
[^2]: Extra task-specific fine-tuning from {cite}`eva02`.
