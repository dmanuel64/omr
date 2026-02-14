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
# ///

import sys
from segment_anything import SamPredictor, sam_model_registry

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

def main() -> None:
    print(sys.executable)
    exit()
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    # predictor.set_image(<your_image>)
    # masks, _, _ = predictor.predict(<input_prompts>)
    print("Hello from export_sam_onnx_model.py!")


if __name__ == "__main__":
    main()
