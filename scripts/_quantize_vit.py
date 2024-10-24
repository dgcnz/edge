import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import timm 
import jsonargparse


def main(batch_size: int = 32, quantize_type: int = "fp8"):
    # model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    model.eval()
    model.cuda()

    training_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    data = iter(training_dataloader)
    images, _ = next(data)

    crit = nn.CrossEntropyLoss()

    # %%
    # Define Calibration Loop for quantization
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


    def calibrate_loop(model):
        # calibrate over the training dataset
        total = 0
        correct = 0
        loss = 0.0
        for data, labels in training_dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = model(data)
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        print("PTQ Loss: {:.5f} Acc: {:.2f}%".format(loss / total, 100 * correct / total))


    # %%
    # Tune the pre-trained model with FP8 and PTQ
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if quantize_type == "int8":
        quant_cfg = mtq.INT8_DEFAULT_CFG
    elif quantize_type == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG
    # PTQ with in-place replacement to quantized modules
    mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # model has FP8 qdq nodes at this point

    # %%
    # Inference
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Load the testing dataset
    testing_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )

    testing_dataloader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )  # set drop_last=True to drop the last incomplete batch for static shape `torchtrt.dynamo.compile()`

    input_tensor = images.cuda()
    with torch.no_grad():
        exp_program = torch.export.export(model, (input_tensor,))
        # with export_torch_mode():
        # with contextlib.nullcontext():
        # Compile the model with Torch-TensorRT Dynamo backend
        # torch.export.export() failed due to RuntimeError: Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()
        # from torch.export._trace import _export

        # exp_program = _export(model, (input_tensor,))
        if quantize_type == "int8":
            enabled_precisions = {torch.int8}
        elif quantize_type == "fp8":
            enabled_precisions = {torch.float8_e4m3fn}
        trt_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=[input_tensor],
            enabled_precisions=enabled_precisions,
            min_block_size=1,
            debug=False,
        )
        # You can also use torch compile path to compile the model with Torch-TensorRT:
        # trt_model = torch.compile(model, backend="tensorrt")

        # Inference compiled Torch-TensorRT model over the testing dataset
        total = 0
        correct = 0
        loss = 0.0
        class_probs = []
        class_preds = []
        for data, labels in testing_dataloader:
            data, labels = data.cuda(), labels.cuda(non_blocking=True)
            out = trt_model(data)
            loss += crit(out, labels)
            preds = torch.max(out, 1)[1]
            class_probs.append([F.softmax(i, dim=0) for i in out])
            class_preds.append(preds)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        test_loss = loss / total
        test_acc = correct / total
        print("Test Loss: {:.5f} Test Acc: {:.2f}%".format(test_loss, 100 * test_acc))


if __name__ == "__main__":
    jsonargparse.CLI(main)