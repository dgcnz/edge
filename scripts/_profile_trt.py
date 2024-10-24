import torch_tensorrt
import torch
import torch_tensorrt
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def load_model(model_path: Path):
    if model_path.suffix == ".ts":
        *_, height, width = model_path.stem.split("_")
        model = torch.jit.load(model_path)
    elif model_path.suffix == ".ep":
        *_, height, width = model_path.stem.split("_")
        model = torch.export.load(model_path).module()
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    return model, int(height), int(width)

def main():
    parser = setup_parser()
    args = parser.parse_args()
    model, height, width = load_model(args.model)
    x = torch.randn(1, 3, height, width).cuda()
    with torch.no_grad():
        # warmup
        for _ in range(5):
            model(x)

        # profile layers
        model(x)

if __name__ == "__main__":
    main()