# ViTs on the Edge

Current results:

| Runtime     |     Model      | Latency | Memory (MB) |
| -------     | -------------- | ------- | ----------- |
| py+trt      | fp16           | 14.043  | 6-500       |
| cpp+trt     | fp16           | 14.600  |  -500       |
| cpp+trt     | fp16+bf16      | 14.655  |             |
| py+trt      | fp32+bf16      | 17.544  | 6-500       |
| cpp+trt     | fp32+bf16      | 18.428  |  -500       |
| py+trt      | fp32           | 37.808  | 6-770       |
| cpp+trt     | fp32           | 37.954  |             |
| py+amp:fp32 | original       | 77.718  | 780-980     |
| py+amp:bf16 | original       | 68.635  | 820-1000    |
| py+amp:fp16 | original       | 67.263  | 820-1000    |
