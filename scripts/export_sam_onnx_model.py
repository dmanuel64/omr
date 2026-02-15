# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "matplotlib>=3.7.5",
#     "onnx>=1.17.0",
#     "onnxruntime>=1.20.1",
#     "opencv-python>=4.13.0.92",
#     "pycocotools>=2.0.7",
#     "torch>=1.7.0",
#     "torchvision>=0.8.0",
#     "segment-anything",
# ]
#
# [tool.uv.sources]
# segment-anything = { git = "https://github.com/facebookresearch/segment-anything.git" }
#
# # Force CUDA wheels for GPU support
# torch = { index = "pytorch-cu130" }
# torchvision = { index = "pytorch-cu130" }
#
# # May need to adjust the CUDA version in the URL based on system configuration
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///

import argparse
from pathlib import Path
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

REPO_ROOT = Path(__file__).parent.parent


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM model to ONNX format")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vit_h", "vit_l", "vit_b"],
        default="vit_h",
        help="Type of the SAM model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "models/sam.onnx"),
        help="Path to save the exported ONNX model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test prediction instead of exporting the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the SAM model checkpoint",
    )
    parser.add_argument("image", type=str, help="Path to a test image for validation")
    args = parser.parse_args()
    sam = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint if Path(args.checkpoint).exists() else None
    )
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
    )
    # Generate masks for the input image
    masks = mask_generator.generate(
        image,
    )
    if args.test:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis("off")
        plt.show()
    else:
        # Export the model to ONNX format
        onnx_model_path = args.output

        onnx_model = SamOnnxModel(sam, return_single_mask=True)

        dynamic_axes = {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }

        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        mask_input_size = [4 * x for x in embed_size]
        dummy_inputs = {
            "image_embeddings": torch.randn(
                1, embed_dim, *embed_size, dtype=torch.float
            ),
            "point_coords": torch.randint(
                low=0, high=1024, size=(1, 5, 2), dtype=torch.float
            ),
            "point_labels": torch.randint(
                low=0, high=4, size=(1, 5), dtype=torch.float
            ),
            "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
            "has_mask_input": torch.tensor([1], dtype=torch.float),
            "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
        }
        output_names = ["masks", "iou_predictions", "low_res_masks"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(onnx_model_path, "wb") as f:
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_inputs.values()),
                    f,
                    export_params=True,
                    verbose=False,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )


if __name__ == "__main__":
    main()
