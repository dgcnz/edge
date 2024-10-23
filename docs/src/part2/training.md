# Training the Decoder

Now we have a working model with a pre-trained backbone, but we still need to train the decoder. Training this model, however, requires quite a bit of compute and time, if we follow the original training recipe: With a NVIDIA A100 GPU (full-node, 80GB), a batch size of 16 uses almost 90% of the memory and takes around 30 hours to train 12 epochs, on full precision.

Training scripts for a SLURM cluster are provided in the `scripts/slurm` folder, but to test the training script locally with a single 16GB GPU, we can use the following command:

```sh
WANDB_MODE=offline python -m scripts.train_net \
    --config-file projects/dino_dinov2/configs/COCO/dino_dinov2_b_12ep.py \
    --num-gpus 1  dataloader.train.total_batch_size=2  \ # reduce batch size to 2
    dataloader.train.num_workers=2 train.amp.enabled=True \ # enable mixed precision training
    model.backbone.net.model_name="vit_small_patch14_dinov2.lvd142m" # use smaller vit
```



::::{grid} 2
:::{grid-item-card} 
:::{figure-md} boxap
<img src="boxap.png" alt="">

Hi
:::
:::
:::{grid-item-card}  
:::{figure-md} loss
<img src="loss.png" alt="">

Hi
:::
:::
::::


base:

bs: 2
full fp
11480MiB /  16380MiB 
Overall training speed: 229 iterations in 0:03:16 (0.8593 s / it)

float16
13021MiB /  16380MiB 
Overall training speed: 231 iterations in 0:02:27 (0.6392 s / it)

small:

full fp
bs:2
Overall training speed: 118 iterations in 0:01:32 (0.7835 s / it)
12771MiB /  16380MiB

float16
bs:2
15143MiB /  16380MiB
Overall training speed: 141 iterations in 0:01:24 (0.6013 s / it)