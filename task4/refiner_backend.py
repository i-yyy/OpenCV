"""CPU-only masked diffusion repair for a LivePortrait head crop."""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2  # Import before torch on Windows; avoids an OpenMP DLL load-order issue.
import numpy as np
from PIL import Image

DEFAULT_REFINER = Path(os.environ.get("TASK4_REFINER", r"D:\task4-models"))
PROMPTS = {
    "human": "clean front-facing Japanese anime portrait, monochrome manga ink line art, grayscale cel shading, symmetric face, two clear matching anime eyes of equal width and height, equal iris size, level eyelids, natural anime nose and mouth, preserve the same character's identity, hairstyle and clothing",
    "animal": "realistic front-facing portrait of the same tabby cat, symmetric natural eyes, same eye color, same nose color, preserve the same striped facial fur pattern, sharp eyes, natural nose, mouth and whiskers, photographic",
}
NEGATIVE = "side view, profile, one eye, closed eye, blurry eye, unequal eyes, mismatched eye size, different iris size, tilted eyelids, deformed, asymmetrical, duplicate, extra eye, extra ear, distorted face, photorealistic, realistic photograph, realistic skin, skin pores, live action, low quality, watermark, girl, woman, feminine, long hair, hair bow, hair ornament"


def facial_repair_mask(height: int, width: int, kind: str, side: str = "all") -> np.ndarray:
    """Soft mask for eyes, nose, mouth and central cheeks in a square head crop."""
    mask = np.zeros((height, width), np.uint8)
    def ellipse(cx: float, cy: float, rx: float, ry: float) -> None:
        cv2.ellipse(mask, (round(cx * width), round(cy * height)), (max(1, round(rx * width)), max(1, round(ry * height))), 0, 0, 360, 255, -1)
    eye_y = 0.43 if kind == "human" else 0.44
    eye_rx, eye_ry = ((0.145, 0.085) if kind == "human" else (0.12, 0.075))
    ellipse(0.34, eye_y, eye_rx, eye_ry)
    ellipse(0.66, eye_y, eye_rx, eye_ry)
    ellipse(0.50, 0.57, 0.19, 0.22)
    ellipse(0.50, 0.68, 0.15, 0.07)
    if kind == "animal": ellipse(0.50, 0.62, 0.24, 0.14)
    if side in ("left", "right"):
        midpoint = width // 2
        if side == "left": mask[:, midpoint:] = 0
        else: mask[:, :midpoint] = 0
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, min(height, width) * 0.018))


def head_generation_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, (width // 2, round(height * 0.46)), (round(width * 0.44), round(height * 0.45)), 0, 0, 360, 255, -1)
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, min(height, width) * 0.025))


class DomainRefinerCPU:
    def __init__(self, model_path: Path = DEFAULT_REFINER):
        self.model_paths = {"human": model_path / "inpaint_anime", "animal": model_path / "inpaint"}
        self.reference_adapter_path = model_path / "ip_adapter"
        self.pipes = {}
        self.reference_adapter_loaded = False
        self.load_seconds = 0.0
        import torch
        self.torch = torch

    def _model_path(self, kind: str) -> Path:
        if kind not in self.model_paths: raise ValueError(f"unsupported kind: {kind}")
        model_path = self.model_paths[kind]
        required = (
            model_path / "model_index.json", model_path / "unet" / "config.json",
            model_path / "vae" / "config.json", model_path / "text_encoder" / "config.json",
            model_path / "tokenizer" / "vocab.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "局部修复模型下载不完整，缺少：\n" + "\n".join(missing) +
                "\n请关闭本程序后联网重新运行：python task4/prepare_models.py --with-refiner"
            )
        return model_path

    def _load(self, kind: str):
        if kind in self.pipes: return self.pipes[kind]
        started = time.perf_counter()
        from diffusers import StableDiffusionInpaintPipeline
        # The cat checkpoint ships its full-precision UNet/VAE as official
        # ``.bin`` files (only its fp16 variants are safetensors).  On CPU we
        # need the full-precision files, so select them explicitly instead of
        # first attempting a missing safetensors filename and emitting a
        # misleading fallback warning.
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            str(self._model_path(kind)), torch_dtype=self.torch.float32,
            safety_checker=None, feature_extractor=None,
            requires_safety_checker=False, local_files_only=True,
            use_safetensors=(kind == "human"),
        )
        pipe.to("cpu")
        # Do not install a sliced attention processor here.  Diffusers 0.32
        # replaces those processors while loading IP-Adapter, and the legacy
        # "max" slicing path constructs SlicedAttnProcessor without its
        # required slice_size on some Windows builds.
        self.pipes[kind], self.load_seconds = pipe, time.perf_counter() - started
        return pipe

    def _enable_reference_adapter(self) -> None:
        """Attach an image-conditioned adapter to the anime inpainting pipeline."""
        if self.reference_adapter_loaded:
            return
        required = (
            self.reference_adapter_path / "models" / "ip-adapter-full-face_sd15.safetensors",
            self.reference_adapter_path / "models" / "image_encoder" / "config.json",
            self.reference_adapter_path / "models" / "image_encoder" / "model.safetensors",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "缺少同角色正脸参考适配器，缺少：\n" + "\n".join(missing) +
                "\n请联网运行：python task4/prepare_models.py --with-reference-adapter"
            )
        pipe = self._load("human")
        try:
            pipe.load_ip_adapter(
                str(self.reference_adapter_path), subfolder="models",
                weight_name="ip-adapter-full-face_sd15.safetensors",
                image_encoder_folder="models/image_encoder", local_files_only=True,
            )
        except OSError as exc:
            # Windows error 1455: the page file is too small to map the 2.5 GB
            # CLIP image encoder alongside the inpainting pipeline.
            if getattr(exc, "winerror", None) == 1455 or "1455" in str(exc):
                raise RuntimeError(
                    "同角色正脸参考无法加载：Windows 可用内存/虚拟内存不足（错误 1455）。\n"
                    "此 CPU 图像编码器约需 2.53 GB，且必须与动漫修复模型同时驻留。\n"
                    "请关闭占内存的软件，并在“系统属性 → 高级 → 性能 → 高级 → 虚拟内存”中启用\n"
                    "“自动管理”，或将页面文件设为至少 16 GB 后重启 Windows；也可取消勾选正脸参考，\n"
                    "继续使用不带参考图的动漫局部修复。"
                ) from exc
            raise
        self.reference_adapter_loaded = True

    def refine(self, crop_rgb: np.ndarray, kind: str, seed: int = 2028, steps: int = 16,
               strength: float = 0.62, side: str = "all", reference_path: Path | None = None,
               reference_scale: float = 0.55) -> tuple[np.ndarray, float]:
        """Inpaint the facial mask only; non-mask pixels remain from the input crop."""
        started = time.perf_counter()
        original = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        mask = facial_repair_mask(512, 512, kind, side)
        pipe = self._load(kind)
        call_kwargs = {}
        if reference_path is not None:
            if kind != "human":
                raise ValueError("同角色正脸参考仅适用于动漫人物")
            if not reference_path.exists():
                raise FileNotFoundError(f"未找到正脸参考图：{reference_path}")
            self._enable_reference_adapter()
            pipe.set_ip_adapter_scale(float(np.clip(reference_scale, 0.0, 1.0)))
            with Image.open(reference_path) as reference_file:
                call_kwargs["ip_adapter_image"] = reference_file.convert("RGB")
        elif self.reference_adapter_loaded and kind == "human":
            pipe.set_ip_adapter_scale(0.0)
        prompt = PROMPTS[kind]
        if kind == "human" and side in ("left", "right"):
            prompt += (
                ", repair only the masked eye and make it match the unmasked eye: "
                "same eye width, eye height, iris radius, eyelid opening and pupil position; "
                "keep the unmasked half unchanged"
            )
        generated = np.asarray(pipe(prompt=prompt, negative_prompt=NEGATIVE,
            image=Image.fromarray(original), mask_image=Image.fromarray(mask),
            strength=float(np.clip(strength, 0.10, 0.85)), num_inference_steps=max(10, int(steps)),
            guidance_scale=5.5, generator=self.torch.Generator(device="cpu").manual_seed(seed), output_type="pil",
            **call_kwargs).images[0].convert("RGB"))
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        output = generated.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
        output = cv2.resize(output.astype(np.uint8), (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return output, time.perf_counter() - started

    def generate_frontal_candidate(self, crop_rgb: np.ndarray, kind: str, seed: int = 2026) -> tuple[np.ndarray, float]:
        """Domain-specific whole-head generative candidate; keeps the outer crop."""
        started = time.perf_counter()
        original = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        mask = head_generation_mask(512, 512)
        pipe = self._load(kind)
        if kind == "human" and self.reference_adapter_loaded:
            pipe.set_ip_adapter_scale(0.0)
        generated = np.asarray(pipe(
            prompt=PROMPTS[kind] + ", centered full frontal view, both eyes visible and symmetrical",
            negative_prompt=NEGATIVE, image=Image.fromarray(original), mask_image=Image.fromarray(mask),
            strength=0.68, num_inference_steps=22, guidance_scale=6.0,
            generator=self.torch.Generator(device="cpu").manual_seed(seed), output_type="pil",
        ).images[0].convert("RGB"))
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        output = generated.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
        output = cv2.resize(output.astype(np.uint8), (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return output, time.perf_counter() - started

    def refine_anime_half(self, crop_rgb: np.ndarray, side: str, seed: int = 2026,
                          strength: float = 0.35, reference_path: Path | None = None,
                          reference_scale: float = 0.55) -> tuple[np.ndarray, float]:
        if side not in ("left", "right"): raise ValueError("side must be 'left' or 'right'")
        return self.refine(
            crop_rgb, "human", seed=seed, strength=strength, side=side,
            reference_path=reference_path, reference_scale=reference_scale,
        )
