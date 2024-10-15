# ViTs on the Edge

Current results:

| Runtime     | Model     | Latency | Memory (MB) |
| ----------- | --------- | ------- | ----------- |
| cpp+trt     | fp16      | 13.808  | 500         |
| py+trt      | fp16      | 13.900  | 500         |
| cpp+trt     | fp16+bf16 | 13.810  | 500         |
| py+trt      | fp16+bf16 | 13.893  | 500         |
| cpp+trt     | fp32+bf16 | 17.366  | 500         |
| py+trt      | fp32+bf16 | 17.439  | 500         |
| cpp+trt     | fp32      | 37.639  | 770         |
| py+trt      | fp32      | 37.738  | 770         |
| py+amp:fp16 | original  | 67.263  | 1000        |
| py+amp:bf16 | original  | 68.635  | 1000        |
| py+fp32     | original  | 77.718  | 980         |
