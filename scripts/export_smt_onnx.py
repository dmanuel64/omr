# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "loguru==0.7.3",
#     "onnx>=1.20.1",
#     "opencv-python==4.11.0.86",
#     "scikit-image==0.25.2",
#     "torch==2.6.0",
#     "torchvision==0.21.0",
#     "transformers==4.49.0",
# ]
# ///


import argparse
import json
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SMT_DIR = REPO_ROOT / "vendor" / "SMT"
sys.path.insert(0, str(SMT_DIR))

from data_augmentation.data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM


class EncoderWrapper(nn.Module):
    def __init__(self, model: SMTModelForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B,1,H,W] float32 [0,1]
        return self.model.forward_encoder(image)  # [B,C,H',W']


class DecoderStepWrapper(nn.Module):
    def __init__(self, model: SMTModelForCausalLM):
        super().__init__()
        self.model = model

    def forward(
        self, encoder_output: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        # encoder_output: [B,C,H',W']
        # tokens: [B,seq] int64
        out = self.model.forward_decoder(
            encoder_output=encoder_output, last_predictions=tokens, return_weights=False
        )
        return out.logits  # [B,seq,vocab]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?", default=str(REPO_ROOT / "models" / "smt"))
    ap.add_argument(
        "--model",
        choices=[
            "antoniorv6/smt-grandstaff",
            "antoniorv6/smt-camera-grandstaff",
            "antoniorv6/smt-string-quartets",
        ],
        default="antoniorv6/smt-grandstaff",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    # Create the destination directory if it does not exist
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check CUDA
    device: str = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but was not found on the device")

    print(f"Loading {args.model} on {device}...")
    model: SMTModelForCausalLM = SMTModelForCausalLM.from_pretrained(args.model).to(
        device
    )
    model.eval()

    # Save vocab for Rust-side decoding
    if isinstance(getattr(model, "i2w", None), dict):
        (dest_dir / "i2w.json").write_text(
            json.dumps(model.i2w, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Wrote i2w.json")
    if isinstance(getattr(model, "w2i", None), dict):
        (dest_dir / "w2i.json").write_text(
            json.dumps(model.w2i, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Wrote w2i.json")

    # Create sample blank image
    maxh = int(model.config.maxh)
    maxw = int(model.config.maxw)
    dummy = np.full((maxh, maxw, 3), 255, dtype=np.uint8)
    image_tensor = convert_img_to_tensor(dummy).unsqueeze(0).to(device)

    # Build BOS token tensor
    bos_id = int(model.w2i["<bos>"])
    tokens = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    # Compute encoder output example once (used for decoder export input signature)
    with torch.no_grad():
        encoder_output = model.forward_encoder(image_tensor)

    print("Exporting ONNX models...")
    # Export encoder
    encoder_path = dest_dir / "smt_encoder.onnx"
    torch.onnx.export(
        EncoderWrapper(model).to(device).eval(),
        (image_tensor,),
        str(encoder_path),
        opset_version=args.opset,
        input_names=["image"],
        output_names=["encoder_output"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "encoder_output": {0: "batch", 2: "enc_h", 3: "enc_w"},
        },
        do_constant_folding=True,
    )
    print(f"Wrote {encoder_path}")
    # Export decoder step
    decoder_path = dest_dir / "smt_decoder_step.onnx"
    torch.onnx.export(
        DecoderStepWrapper(model).to(device).eval(),
        (encoder_output, tokens),
        str(decoder_path),
        opset_version=args.opset,
        input_names=["encoder_output", "tokens"],
        output_names=["logits"],
        dynamic_axes={
            "encoder_output": {0: "batch", 2: "enc_h", 3: "enc_w"},
            "tokens": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        do_constant_folding=True,
    )
    print(f"Wrote {decoder_path}")


if __name__ == "__main__":
    main()
