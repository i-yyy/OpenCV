"""Minimal, CPU-only ONNX inference backend for interactive head rotation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def rotation_matrix(pitch: np.ndarray, yaw: np.ndarray, roll: np.ndarray) -> np.ndarray:
    pitch, yaw, roll = [np.asarray(v, np.float32).reshape(-1, 1) * np.pi / 180 for v in (pitch, yaw, roll)]
    one, zero = np.ones_like(pitch), np.zeros_like(pitch)
    rx = np.concatenate((one, zero, zero, zero, np.cos(pitch), -np.sin(pitch), zero, np.sin(pitch), np.cos(pitch)), 1).reshape(-1, 3, 3)
    ry = np.concatenate((np.cos(yaw), zero, np.sin(yaw), zero, one, zero, -np.sin(yaw), zero, np.cos(yaw)), 1).reshape(-1, 3, 3)
    rz = np.concatenate((np.cos(roll), -np.sin(roll), zero, np.sin(roll), np.cos(roll), zero, zero, zero, one), 1).reshape(-1, 3, 3)
    return np.transpose(rz @ (ry @ rx), (0, 2, 1)).astype(np.float32)


def headpose_to_degree(pred: np.ndarray) -> np.ndarray:
    if pred.ndim > 1 and pred.shape[1] == 66:
        p = np.exp(pred - pred.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return (p * np.arange(66, dtype=np.float32)).sum(axis=1) * 3 - 97.5
    return pred.reshape(-1)


class CpuSession:
    def __init__(self, path: Path, threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.inputs = self.session.get_inputs()

    def run(self, *arrays: np.ndarray) -> list[np.ndarray]:
        feed = {spec.name: np.asarray(value, np.float32) for spec, value in zip(self.inputs, arrays)}
        return self.session.run(None, feed)


@dataclass
class SourceState:
    original_rgb: np.ndarray
    crop_rgb: np.ndarray
    box: tuple[int, int, int, int]
    feature: np.ndarray
    kp: np.ndarray
    exp: np.ndarray
    scale: np.ndarray
    translation: np.ndarray
    pitch: float
    yaw: float
    roll: float
    source_kp: np.ndarray
    read_seconds: float
    prepare_seconds: float


class LivePortraitCPU:
    """Loads one domain at a time and always renders from the cached source."""

    def __init__(self, models_root: Path, kind: str, threads: int = 4):
        if kind not in ("human", "animal"):
            raise ValueError("kind must be human or animal")
        folder = models_root / kind
        paths = {
            "appearance": folder / "appearance_feature_extractor.onnx",
            "motion": folder / "motion_extractor.onnx",
            "warp": folder / "warping_spade.onnx",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError("缺少模型文件：\n" + "\n".join(missing) + "\n请先运行 prepare_models.py")
        started = time.perf_counter()
        self.appearance = CpuSession(paths["appearance"], threads)
        self.motion = CpuSession(paths["motion"], threads)
        self.warp = CpuSession(paths["warp"], threads)
        self.load_seconds = time.perf_counter() - started
        self.state: SourceState | None = None

    @staticmethod
    def _network_input(rgb: np.ndarray) -> np.ndarray:
        crop = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        return np.transpose(crop.astype(np.float32) / 255.0, (2, 0, 1))[None]

    def prepare(self, image_path: Path, box: tuple[int, int, int, int] | None = None) -> SourceState:
        started = time.perf_counter()
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"无法读取图像：{image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        read_seconds = time.perf_counter() - started
        h, w = rgb.shape[:2]
        if box is None:
            side = min(h, w)
            x0, y0 = (w - side) // 2, (h - side) // 2
            box = (x0, y0, x0 + side, y0 + side)
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 < 32 or y1 - y0 < 32:
            raise ValueError("裁剪框太小")
        crop = rgb[y0:y1, x0:x1]
        inp = self._network_input(crop)
        feature = self.appearance.run(inp)[0]
        pitch, yaw, roll, trans, exp, scale, kp = self.motion.run(inp)
        pitch, yaw, roll = map(headpose_to_degree, (pitch, yaw, roll))
        kp = kp.reshape(kp.shape[0], -1, 3)
        exp = exp.reshape(exp.shape[0], -1, 3)
        rot = rotation_matrix(pitch, yaw, roll)
        source_kp = scale[..., None] * (kp @ rot + exp)
        source_kp[:, :, :2] += trans[:, None, :2]
        self.state = SourceState(rgb, crop, (x0, y0, x1, y1), feature, kp, exp, scale, trans,
                                 float(pitch[0]), float(yaw[0]), float(roll[0]), source_kp,
                                 read_seconds, time.perf_counter() - started)
        return self.state

    def render(self, target_yaw: float, pitch_correction: float = 0.55,
               roll_correction: float = 0.70) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Render a target yaw while gently levelling pitch and roll.

        A full pitch correction often distorts the forehead and chin, so callers
        pass correction strengths in [0, 1], rather than forcing all axes to 0.
        """
        if self.state is None:
            raise RuntimeError("尚未准备原图")
        s = self.state
        started = time.perf_counter()
        pitch_correction = float(np.clip(pitch_correction, 0.0, 1.0))
        roll_correction = float(np.clip(roll_correction, 0.0, 1.0))
        target_pitch = s.pitch * (1.0 - pitch_correction)
        target_roll = s.roll * (1.0 - roll_correction)
        rot = rotation_matrix(np.array([target_pitch]), np.array([target_yaw]), np.array([target_roll]))
        target_kp = s.scale[..., None] * (s.kp @ rot + s.exp)
        target_kp[:, :, :2] += s.translation[:, None, :2]
        # The ONNX graph input order is feature, driving keypoints, source keypoints.
        output = self.warp.run(s.feature, target_kp, s.source_kp)[0]
        crop = np.clip(np.transpose(output[0], (1, 2, 0)), 0, 1)
        crop = (crop * 255).astype(np.uint8)
        result = self.compose(crop)
        return crop, result, time.perf_counter() - started, target_pitch, target_roll

    def compose(self, crop: np.ndarray) -> np.ndarray:
        """Feather a generated 512-square crop back onto the untouched source image."""
        if self.state is None:
            raise RuntimeError("尚未准备原图")
        s = self.state
        x0, y0, x1, y1 = s.box
        resized = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)
        result = s.original_rgb.copy()
        # Feather only the outer edge. This retains the original background and avoids a hard square seam.
        mh, mw = resized.shape[:2]
        mask = np.ones((mh, mw), np.float32)
        feather = max(4, min(mh, mw) // 24)
        mask[:feather] *= np.linspace(0, 1, feather)[:, None]
        mask[-feather:] *= np.linspace(1, 0, feather)[:, None]
        mask[:, :feather] *= np.linspace(0, 1, feather)[None, :]
        mask[:, -feather:] *= np.linspace(1, 0, feather)[None, :]
        roi = result[y0:y1, x0:x1]
        result[y0:y1, x0:x1] = (resized * mask[..., None] + roi * (1 - mask[..., None])).astype(np.uint8)
        return result
