"""Native FLUX.2 Klein 4B image-edit client for the local ComfyUI service."""
from __future__ import annotations

import json
import time
import uuid
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np



class LocalComfyClient:
    """Minimal HTTP client shared by the local FLUX.2 services."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, path: str, data: dict | None = None, timeout: float = 30.0) -> bytes:
        body = None if data is None else json.dumps(data).encode("utf-8")
        request = Request(
            self.base_url + path, data=body,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except URLError as exc:
            raise RuntimeError(f"Local FLUX.2 service is unavailable at {self.base_url}") from exc

    def _upload_reference(self, source_rgb: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Unable to encode the source image as PNG")
        boundary = "----task4Flux2Boundary"
        chunks = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                f"filename=\"task4_flux2_reference_{uuid.uuid4().hex}.png\"\r\n"
                "Content-Type: image/png\r\n\r\n"
            ).encode(),
            encoded.tobytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        request = Request(
            self.base_url + "/upload/image", data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urlopen(request, timeout=120.0) as response:
                uploaded = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError("Unable to upload the source image to local FLUX.2") from exc
        name = uploaded.get("name")
        if not name:
            raise RuntimeError("ComfyUI did not return an uploaded image name: " + str(uploaded))
        return name


class Flux2KleinGPU(LocalComfyClient):
    """Use FLUX.2's native ReferenceLatent image-edit conditioning."""

    def ensure_ready(self) -> None:
        info = json.loads(self._request("/object_info").decode("utf-8"))
        required = (
            "UNETLoader", "CLIPLoader", "VAELoader", "ReferenceLatent",
            "EmptyFlux2LatentImage", "Flux2Scheduler", "CFGGuider",
            "RandomNoise", "KSamplerSelect", "SamplerCustomAdvanced",
        )
        missing = [name for name in required if name not in info]
        if missing:
            raise RuntimeError("FLUX.2 service lacks nodes: " + ", ".join(missing))
        expected = {
            "flux-2-klein-4b-fp8.safetensors": info["UNETLoader"]["input"]["required"]["unet_name"][0],
            "qwen_3_4b.safetensors": info["CLIPLoader"]["input"]["required"]["clip_name"][0],
            "flux2-vae.safetensors": info["VAELoader"]["input"]["required"]["vae_name"][0],
        }
        absent = [name for name, choices in expected.items() if name not in choices]
        if absent:
            raise RuntimeError("FLUX.2 model files are not visible to ComfyUI: " + ", ".join(absent))

    def _workflow(self, kind: str, seed: int, source_name: str, width: int, height: int) -> dict:
        # Mirrors the official FLUX.2 Klein 4B distilled image-edit workflow:
        # source VAE latent is used through ReferenceLatent for both the
        # positive and zeroed-negative conditions, while sampling begins fresh.
        return {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b-fp8.safetensors", "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
            "4": {"class_type": "LoadImage", "inputs": {"image": source_name}},
            "5": {"class_type": "ImageScale", "inputs": {"image": ["4", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}},
            "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": self._prompt(kind)}},
            "8": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
            "9": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]}},
            "10": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["6", 0]}},
            "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "12": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
            "13": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["9", 0], "negative": ["10", 0], "cfg": 1.0}},
            "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "15": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": width, "height": height}},
            "16": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["12", 0], "guider": ["13", 0], "sampler": ["14", 0], "sigmas": ["15", 0], "latent_image": ["11", 0]}},
            "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}},
            "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": "task4_flux2_klein_edit"}},
        }

    def generate(self, source_rgb: np.ndarray, kind: str, seed: int,
                 poll_seconds: float = 2.0, timeout_seconds: float = 3600.0) -> tuple[np.ndarray, float]:
        self.ensure_ready()
        started = time.perf_counter()
        source_name = self._upload_reference(source_rgb)
        height, width = source_rgb.shape[:2]
        # 512 px is a conservative initial resolution for the 8 GB laptop GPU.
        scale = min(512 / width, 512 / height)
        width = max(16, int(round(width * scale / 16)) * 16)
        height = max(16, int(round(height * scale / 16)) * 16)
        payload = json.loads(self._request("/prompt", {"prompt": self._workflow(kind, seed, source_name, width, height)}).decode("utf-8"))
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("ComfyUI rejected the FLUX.2 job: " + json.dumps(payload, ensure_ascii=False))
        deadline = started + timeout_seconds
        while time.perf_counter() < deadline:
            history = json.loads(self._request(f"/history/{prompt_id}").decode("utf-8"))
            job = history.get(prompt_id)
            if job:
                if job.get("status", {}).get("status_str") == "error":
                    messages = job.get("status", {}).get("messages", [])
                    raise RuntimeError("FLUX.2 generation failed: " + str(messages[-1] if messages else job))
                images = job.get("outputs", {}).get("18", {}).get("images", [])
                if images:
                    image = images[0]
                    query = urlencode({"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")})
                    encoded = np.frombuffer(self._request("/view?" + query, timeout=120.0), np.uint8)
                    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError("ComfyUI returned an unreadable FLUX.2 image")
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), time.perf_counter() - started
            time.sleep(poll_seconds)
        raise TimeoutError("FLUX.2 did not finish within 60 minutes; inspect the 8191 service terminal.")
