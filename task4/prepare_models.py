"""Download the minimal FasterLivePortrait ONNX weights used by task4."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Xet creates temporary files at the drive root on some Windows setups. Plain HTTP
# is slower but works with a folder-scoped D: permission and supports resuming.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download


def download_with_retries(download, label: str, attempts: int = 5):
    """Retry transient HTTP stream failures; Hugging Face reuses partial cache data."""
    for attempt in range(1, attempts + 1):
        try:
            return download()
        except (OSError, ConnectionError) as exc:
            if attempt == attempts:
                raise
            delay = min(60, 3 * 2 ** (attempt - 1))
            print(f"[{label}] download interrupted ({exc.__class__.__name__}); retrying {attempt}/{attempts} in {delay}s...", flush=True)
            time.sleep(delay)


REPO_ID = "warmshao/FasterLivePortrait"
FILES = {
    "human": (
        "liveportrait_onnx/appearance_feature_extractor.onnx",
        "liveportrait_onnx/motion_extractor.onnx",
        "liveportrait_onnx/warping_spade.onnx",
    ),
    "animal": (
        "liveportrait_animal_onnx/appearance_feature_extractor.onnx",
        "liveportrait_animal_onnx/motion_extractor.onnx",
        "liveportrait_animal_onnx/warping_spade.onnx",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--only", choices=("human", "animal", "all"), default="all")
    parser.add_argument("--with-refiner", action="store_true", help="also download local inpainting refiners")
    parser.add_argument("--refiner-kind", choices=("human", "animal", "all"), default="all")
    parser.add_argument("--with-reference-adapter", action="store_true",
                        help="download the CPU same-character image-reference adapter for anime repair")
    parser.add_argument("--refiner-output", type=Path, default=Path(r"D:\task4-models"))
    args = parser.parse_args()
    kinds = FILES if args.only == "all" else {args.only: FILES[args.only]}
    args.output.mkdir(parents=True, exist_ok=True)
    for kind, files in kinds.items():
        target = args.output / kind
        target.mkdir(parents=True, exist_ok=True)
        for remote_name in files:
            print(f"[{kind}] downloading {remote_name}", flush=True)
            cached = download_with_retries(lambda: hf_hub_download(REPO_ID, remote_name), f"{kind}/{Path(remote_name).name}")
            destination = target / Path(remote_name).name
            if destination.exists():
                destination.unlink()
            try:
                destination.hardlink_to(cached)
            except OSError:
                import shutil
                shutil.copy2(cached, destination)
            print(f"  -> {destination}", flush=True)
    if args.with_refiner:
        from huggingface_hub import snapshot_download
        # This is an inpainting checkpoint (not an ordinary img2img checkpoint):
        # it accepts image + mask, allowing us to preserve hair, clothing and background.
        # Keep only files used by this CPU pipeline.  In particular, avoid
        # downloading duplicate fp16 variants and the unused safety checker.
        patterns = [
            "model_index.json", "scheduler/scheduler_config.json", "tokenizer/*",
            "text_encoder/config.json", "text_encoder/pytorch_model.bin", "text_encoder/model.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.bin", "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.bin", "vae/diffusion_pytorch_model.safetensors",
            "feature_extractor/preprocessor_config.json",
        ]
        if args.refiner_kind in ("animal", "all"):
            cat_dir = args.refiner_output / "inpaint"
            print("[refiner] downloading runwayml/stable-diffusion-inpainting for cat local repair (about 5 GB)", flush=True)
            download_with_retries(lambda: snapshot_download("runwayml/stable-diffusion-inpainting", local_dir=cat_dir, max_workers=1, allow_patterns=patterns), "cat-inpaint")
            print(f"  -> {cat_dir}", flush=True)
        if args.refiner_kind in ("human", "all"):
            anime_inpaint_dir = args.refiner_output / "inpaint_anime"
            print("[refiner] downloading Sanster/anything-4.0-inpainting for anime local repair (about 2.8 GB)", flush=True)
            download_with_retries(lambda: snapshot_download("Sanster/anything-4.0-inpainting", local_dir=anime_inpaint_dir, max_workers=1, allow_patterns=patterns), "anime-inpaint")
            print(f"  -> {anime_inpaint_dir}", flush=True)
    if args.with_reference_adapter:
        from huggingface_hub import snapshot_download
        adapter_dir = args.refiner_output / "ip_adapter"
        print("[reference] downloading h94/IP-Adapter image encoder and SD1.5 face adapter (about 2.6 GB)", flush=True)
        download_with_retries(
            lambda: snapshot_download(
                "h94/IP-Adapter", local_dir=adapter_dir, max_workers=1,
                allow_patterns=[
                    "models/ip-adapter-full-face_sd15.safetensors",
                    "models/image_encoder/config.json",
                    "models/image_encoder/model.safetensors",
                    "models/image_encoder/preprocessor_config.json",
                ],
            ),
            "reference-adapter",
        )
        print(f"  -> {adapter_dir}", flush=True)
    print("Models are ready. Inference itself can now run fully offline.")


if __name__ == "__main__":
    main()
