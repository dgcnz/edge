# Problem Definition

TODO
- [ ] Rewrite slides in text format

---

Training a model from scratch requires a labeled dataset that covers the target domain (🌧️, 🌓, 👁️, …).

Curating such a dataset is hard and expensive.

However, what if we could rely on a model pre-trained on 141 million images?

---

Recent paradigm shift, from deep learning to foundation models (Bommasani, 2022):
DL: Train a model for a specific task and dataset (e.g. object detection of blood cells)

FOMO: Pre-train a large model with a huge unlabeled dataset on a generic task (e.g. predicting missing patches on an image) and then adapt it for downstream tasks
Adaptations include: fine-tuning, decoder training, distillation, quantization, sparsification, etc.

---

According to {cite}`mcip`, practitioners at Apple do the following when asked to deploy a model to some edge device.
Find a feasibility model A.
Find a smaller model B equivalent to A that is suitable for the device.
Compress model B to reach production-ready model C.


:::{figure-md} apple_practice
<img src="apple_practice.png" alt="">

Hi
:::