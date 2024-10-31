# Conclusions

In terms of results, the main takeaway is that, if we consider the {ref}`sota`,  we've achieved 15 ms (or equivalently, 67 FPS) which places us near YOLOv8-X (59 FPS). Although this is not a fair comparison, as YOLOv8-X is ran on a slower T4 GPU, this still leaves us in a good position for real-time applications. Furthermore, we've manged to half the memory usage of the model (from 1GB to 500MB). 

The bulk of this work was spent in optimizing by compilation procedures, which we've thoroughly documented in the two case studies of {ref}`part2:compilingmodel`. However, much work can still be done in the optimization of the model itself (quantization, structured pruning, etc) and in doing architecture search, as described in {ref}`part1:objectives`.

Although not documented in these tutorials, we've tried quantization techniques that were unsuccessful or whose results were not significant enough to be included in this report. Scripts for these experiments are available in the repository and future work needs to be done in this direction if we aim to deploy these models not only on small edge GPUs but also on mobile devices.

