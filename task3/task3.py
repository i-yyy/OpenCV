from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageGrab, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:
    raise SystemExit("Tkinter is required to run this desktop application.") from exc

import cv2  # type: ignore


APP_DIR = Path(__file__).resolve().parent
BACKGROUND_DIR = APP_DIR / "background"
FOREGROUND_DIR = APP_DIR / "foreground"
RESULT_DIR = APP_DIR / "result"

UI_FONT = "Microsoft YaHei"
BG = "#edf6ff"
PANEL = "#ffffff"
INK = "#202833"
MUTED = "#667085"
LINE = "#dbe7f5"
BLUE = "#2563eb"
GREEN = "#079455"
ORANGE = "#dc6803"
SOFT_BLUE = "#f5faff"
CARD_BG = "#fbfdff"
BUTTON_BG = "#ffffff"
ACTIVE_BLUE = "#2f80ed"
CANVAS_TARGET_WIDTH = 760
CANVAS_MIN_WIDTH = 640
CANVAS_TARGET_HEIGHT = 470
PREVIEW_CANVAS_HEIGHT = 155
WINDOW_EXTRA_WIDTH = 760
WINDOW_EXTRA_HEIGHT = 245


@dataclass
class DisplayGeometry:
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    width: int = 1
    height: int = 1


@dataclass
class CylinderSurface:
    box: tuple[int, int, int, int]
    top: np.ndarray
    bottom: np.ndarray
    top_arc_factor: float = 1.0
    bottom_arc_factor: float = 1.0


@dataclass
class UndoState:
    result_rgb: np.ndarray | None
    current_mask: np.ndarray | None
    brush_map: np.ndarray | None
    points: list[tuple[float, float]]
    interaction_count: int
    wrap: int
    cover: int
    light: int
    texture: int
    opacity: int
    feather: int
    brightness_adjust: int
    hue_shift: int
    temperature: int
    saturation: int
    contrast: int
    blend: str
    brightness: bool
    top_arc_factor: float
    bottom_arc_factor: float


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").save(path)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in suffixes)


def order_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = np.asarray(points, dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(sums))]
    br = pts[int(np.argmax(sums))]
    tr = pts[int(np.argmax(diffs))]
    bl = pts[int(np.argmin(diffs))]
    return [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1])), (float(br[0]), float(br[1])), (float(bl[0]), float(bl[1]))]


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    dx = xs - x0
    dy = ys - y0

    wa = (1.0 - dx) * (1.0 - dy)
    wb = dx * (1.0 - dy)
    wc = (1.0 - dx) * dy
    wd = dx * dy

    if image.ndim == 2:
        return (
            image[y0, x0] * wa
            + image[y0, x1] * wb
            + image[y1, x0] * wc
            + image[y1, x1] * wd
        )

    return (
        image[y0, x0] * wa[..., None]
        + image[y0, x1] * wb[..., None]
        + image[y1, x0] * wc[..., None]
        + image[y1, x1] * wd[..., None]
    )


def warp_perspective(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: tuple[int, int],
    is_mask: bool = False,
) -> np.ndarray:
    matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border_mode = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REPLICATE
    return cv2.warpPerspective(image, matrix, output_size, flags=interpolation, borderMode=border_mode)


def warp_soft_mask(
    mask: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
    return cv2.warpPerspective(mask, matrix, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def bleed_edge_colors(rgb: np.ndarray, alpha: np.ndarray, iterations: int = 16) -> np.ndarray:
    output = rgb.copy()
    filled = np.asarray(alpha) > 0
    if not np.any(filled) or np.all(filled):
        return output

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    filled_u8 = filled.astype(np.uint8)
    for _ in range(iterations):
        expanded = cv2.dilate(filled_u8, kernel, iterations=1)
        ring = (expanded > 0) & (filled_u8 == 0)
        if not np.any(ring):
            break
        for channel_index in range(3):
            source = output[:, :, channel_index].copy()
            source[filled_u8 == 0] = 0
            output[:, :, channel_index][ring] = cv2.dilate(source, kernel, iterations=1)[ring]
        filled_u8[ring] = 1
    return output


def make_foreground_mask(path: Path, rgb: np.ndarray) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA"):
            alpha = np.array(image.convert("RGBA"))[:, :, 3]
            if int(alpha.max()) > int(alpha.min()):
                return alpha.astype(np.uint8)

    arr = rgb.astype(np.int16)
    near_black = np.max(arr, axis=2) < 18
    mask = np.where(~near_black, 255, 0).astype(np.uint8)
    if np.count_nonzero(mask) < mask.size * 0.02:
        mask[:, :] = 255
    mask = clean_mask(mask, kernel_size=5)
    return fill_mask_holes(mask)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 20, 255, 0).astype(np.uint8)
    flood = binary.copy()
    height, width = flood.shape
    fill = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, fill, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes).astype(np.uint8)


def crop_to_mask(
    foreground: np.ndarray,
    mask: np.ndarray,
    padding: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    active = mask > 20
    if not np.any(active):
        return foreground, mask
    ys, xs = np.where(active)
    height, width = mask.shape
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(width - 1, int(xs.max()) + padding)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(height - 1, int(ys.max()) + padding)
    return foreground[y1 : y2 + 1, x1 : x2 + 1].copy(), mask[y1 : y2 + 1, x1 : x2 + 1].copy()


def fit_foreground_to_canvas(
    foreground: np.ndarray,
    mask: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    padding_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    cropped_rgb, cropped_mask = crop_to_mask(foreground, mask)
    src_h, src_w = cropped_rgb.shape[:2]
    inner_w = max(1, int(canvas_width * (1.0 - padding_ratio * 2.0)))
    inner_h = max(1, int(canvas_height * (1.0 - padding_ratio * 2.0)))
    scale = min(inner_w / max(1, src_w), inner_h / max(1, src_h))
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))

    rgb_img = Image.fromarray(cropped_rgb, mode="RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_img = Image.fromarray(cropped_mask, mode="L").resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas_rgb = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    x = (canvas_width - new_w) // 2
    y = (canvas_height - new_h) // 2
    canvas_rgb[y : y + new_h, x : x + new_w] = np.array(rgb_img, dtype=np.uint8)
    canvas_mask[y : y + new_h, x : x + new_w] = np.array(mask_img, dtype=np.uint8)
    return canvas_rgb, canvas_mask


def fit_foreground_cover_canvas(
    foreground: np.ndarray,
    mask: np.ndarray,
    canvas_width: int,
    canvas_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    cropped_rgb, cropped_mask = crop_to_mask(foreground, mask, padding=2)
    src_h, src_w = cropped_rgb.shape[:2]
    scale = max(canvas_width / max(1, src_w), canvas_height / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    rgb_img = Image.fromarray(cropped_rgb, mode="RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_img = Image.fromarray(cropped_mask, mode="L").resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = max(0, (new_w - canvas_width) // 2)
    top = max(0, (new_h - canvas_height) // 2)
    right = min(new_w, left + canvas_width)
    bottom = min(new_h, top + canvas_height)

    rgb_crop = np.array(rgb_img.crop((left, top, right, bottom)), dtype=np.uint8)
    mask_crop = np.array(mask_img.crop((left, top, right, bottom)), dtype=np.uint8)

    canvas_rgb = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    h, w = rgb_crop.shape[:2]
    x = (canvas_width - w) // 2
    y = (canvas_height - h) // 2
    canvas_rgb[y : y + h, x : x + w] = rgb_crop
    canvas_mask[y : y + h, x : x + w] = mask_crop
    return canvas_rgb, canvas_mask


def fit_foreground_stretch_canvas(
    foreground: np.ndarray,
    mask: np.ndarray,
    canvas_width: int,
    canvas_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    cropped_rgb, cropped_mask = crop_to_mask(foreground, mask, padding=2)
    rgb_img = Image.fromarray(cropped_rgb, mode="RGB").resize(
        (canvas_width, canvas_height),
        Image.Resampling.LANCZOS,
    )
    mask_img = Image.fromarray(cropped_mask, mode="L").resize(
        (canvas_width, canvas_height),
        Image.Resampling.LANCZOS,
    )
    resized_rgb = np.array(rgb_img, dtype=np.uint8)
    resized_mask = np.array(mask_img, dtype=np.uint8)
    # 缩放会在透明区域重新产生黑色 RGB，先扩散有效前景颜色再交给柱面 remap。
    resized_rgb = bleed_edge_colors(resized_rgb, resized_mask, iterations=24)
    return resized_rgb, resized_mask


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    binary = np.where(mask > 20, 255, 0).astype(np.uint8)
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary.astype(np.uint8)


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    base = np.clip(mask, 0, 255).astype(np.uint8)
    if radius <= 0:
        return base
    binary = np.where(base > 20, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return base
    image = Image.fromarray(base, mode="L")
    blurred = np.array(image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    output = blurred
    output[distance >= radius] = 255
    output[(binary == 0) & (output < 1)] = 0
    return np.clip(output, 0, 255).astype(np.uint8)


def adjust_brightness(foreground: np.ndarray, percent: int) -> np.ndarray:
    factor = float(np.clip(percent, 40, 180)) / 100.0
    adjusted = foreground.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_foreground_color_adjustments(
    foreground: np.ndarray,
    hue_shift: int = 0,
    temperature: int = 0,
    tint: int = 0,
    saturation: int = 100,
    vibrance: int = 0,
    lightness: int = 0,
    exposure: int = 0,
    contrast: int = 100,
    highlights: int = 0,
    shadows: int = 0,
) -> np.ndarray:
    if (
        hue_shift == 0
        and temperature == 0
        and tint == 0
        and saturation == 100
        and vibrance == 0
        and lightness == 0
        and exposure == 0
        and contrast == 100
        and highlights == 0
        and shadows == 0
    ):
        return foreground.astype(np.uint8)

    rgb = foreground.astype(np.float32) / 255.0

    exposure_gain = 2.0 ** (float(np.clip(exposure, -100, 100)) / 100.0)
    rgb *= exposure_gain

    temp = float(np.clip(temperature, -100, 100)) / 100.0
    tint_amount = float(np.clip(tint, -100, 100)) / 100.0
    gains = np.array(
        [
            1.0 + 0.20 * temp + 0.08 * tint_amount,
            1.0 - 0.08 * abs(temp) - 0.16 * tint_amount,
            1.0 - 0.20 * temp + 0.08 * tint_amount,
        ],
        dtype=np.float32,
    )
    rgb *= gains[None, None, :]

    contrast_gain = float(np.clip(contrast, 20, 220)) / 100.0
    rgb = (rgb - 0.5) * contrast_gain + 0.5

    luma = np.clip(rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32), 0.0, 1.0)
    shadow_amount = float(np.clip(shadows, -100, 100)) / 100.0
    if abs(shadow_amount) > 1e-6:
        weight = (1.0 - luma) ** 2.0
        if shadow_amount > 0:
            rgb += (1.0 - rgb) * (0.42 * shadow_amount * weight[..., None])
        else:
            rgb *= 1.0 + 0.36 * shadow_amount * weight[..., None]

    highlight_amount = float(np.clip(highlights, -100, 100)) / 100.0
    if abs(highlight_amount) > 1e-6:
        weight = luma**2.0
        if highlight_amount > 0:
            rgb += (1.0 - rgb) * (0.36 * highlight_amount * weight[..., None])
        else:
            rgb *= 1.0 + 0.42 * highlight_amount * weight[..., None]

    rgb += float(np.clip(lightness, -100, 100)) / 100.0 * 0.32
    rgb = np.clip(rgb, 0.0, 1.0)

    hsv = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + float(np.clip(hue_shift, -180, 180)) / 2.0) % 180.0
    sat = hsv[:, :, 1] / 255.0
    sat_scale = float(np.clip(saturation, 0, 300)) / 100.0
    vibrance_amount = float(np.clip(vibrance, -100, 100)) / 100.0
    vibrance_scale = 1.0 + 0.85 * vibrance_amount * (1.0 - sat)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale * vibrance_scale, 0.0, 255.0)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def match_brightness(foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    active = alpha > 20
    if not np.any(active):
        return foreground
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    fg_luma = np.mean(fg[active] @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
    bg_luma = np.mean(bg[active] @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
    if fg_luma <= 1:
        return foreground
    ratio = float(np.clip(bg_luma / fg_luma, 0.55, 1.45))
    adjusted = np.clip(fg * ratio, 0, 255)
    return adjusted.astype(np.uint8)


def adapt_edge_colors_to_background(
    foreground: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """让羽化带的颜色逐渐取自当前位置背景，降低透明边缘的色晕。"""
    alpha_float = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    if not np.any(alpha_float < 0.98):
        return foreground.astype(np.uint8)

    # 仅处理羽化边缘，主体区域保留原始前景颜色。
    edge_progress = np.clip((alpha_float - 0.04) / 0.62, 0.0, 1.0)
    edge_progress = edge_progress * edge_progress * (3.0 - 2.0 * edge_progress)
    background_weight = 1.0 - edge_progress
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    adapted = fg * (1.0 - background_weight[..., None]) + bg * background_weight[..., None]
    return np.clip(adapted, 0, 255).astype(np.uint8)


def apply_edge_brush(
    result: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    brush_map: np.ndarray | None,
) -> np.ndarray:
    """将笔刷强度限制在前景边缘，并平滑混合前景与背景。"""
    if brush_map is None or not np.any(brush_map > 0):
        return result

    binary = np.where(mask > 20, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return result
    inside_distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
    distance_to_edge = np.where(binary > 0, inside_distance, outside_distance)
    edge_width = 24.0
    edge = np.exp(-distance_to_edge / edge_width).astype(np.float32)
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=2.0, sigmaY=2.0)
    strength = np.clip(brush_map.astype(np.float32) / 255.0, 0.0, 1.0) * edge
    if not np.any(strength > 0.001):
        return result

    # 用局部平滑后的颜色参与混合，避免笔刷留下新的硬边。
    smooth_result = cv2.GaussianBlur(result, (0, 0), sigmaX=2.2, sigmaY=2.2)
    mixed = smooth_result.astype(np.float32) * 0.42 + background.astype(np.float32) * 0.58
    amount = np.clip(strength * 0.92, 0.0, 0.92)[..., None]
    output = result.astype(np.float32) * (1.0 - amount) + mixed * amount
    return np.clip(output, 0, 255).astype(np.uint8)


def alpha_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    opacity: float,
    feather: int,
    brightness_match: bool,
    brightness_adjust: int,
) -> np.ndarray:
    alpha = feather_mask(mask, feather).astype(np.float32) / 255.0
    alpha *= float(np.clip(opacity, 0.0, 1.0))
    prepared = foreground.astype(np.uint8)
    if brightness_match:
        prepared = match_brightness(prepared, background, (alpha * 255).astype(np.uint8))
    prepared = adjust_brightness(prepared, brightness_adjust)
    result = background.astype(np.float32) * (1.0 - alpha[..., None]) + prepared.astype(np.float32) * alpha[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def try_poisson_blend(background: np.ndarray, foreground: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    active = mask > 10
    if not np.any(active):
        return None
    ys, xs = np.where(active)
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    try:
        src_bgr = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        blend_mask = np.where(mask > 10, 255, 0).astype(np.uint8)
        cloned = cv2.seamlessClone(src_bgr, dst_bgr, blend_mask, center, cv2.NORMAL_CLONE)
        return cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def try_gradient_edge_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    base_result: np.ndarray,
    mask: np.ndarray,
    feather: int,
) -> np.ndarray | None:
    """只在 mask 的边缘环带执行梯度域融合，前景内部保持普通融合。"""
    binary = np.where(mask > 10, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return None

    kernel_size = max(3, min(11, int(feather) * 2 + 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    edge_mask = np.where((dilated > 0) & (eroded == 0), 255, 0).astype(np.uint8)
    if np.count_nonzero(edge_mask) < 20:
        return base_result

    ys, xs = np.where(edge_mask > 0)
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    try:
        edge_weight = cv2.GaussianBlur(edge_mask, (0, 0), sigmaX=max(1.0, feather / 2.0)).astype(np.float32) / 255.0
        # 高斯只用于边缘环带内部的柔化，环带之外严格不参与梯度融合。
        edge_weight[edge_mask == 0] = 0.0
        edge_weight *= 0.82
        src_bgr = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        cloned = cv2.seamlessClone(src_bgr, dst_bgr, edge_mask.copy(), center, cv2.NORMAL_CLONE)
        cloned_rgb = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)
        result = base_result.astype(np.float32) * (1.0 - edge_weight[..., None])
        result += cloned_rgb.astype(np.float32) * edge_weight[..., None]
        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception:
        return None


def add_screen_finish(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    active = mask > 10
    if not np.any(active):
        return image
    output = image.astype(np.float32)
    height, width = mask.shape
    yy, xx = np.mgrid[0:height, 0:width]
    gloss = np.clip(1.0 - (xx / max(1, width) * 0.45 + yy / max(1, height) * 0.28), 0.58, 1.0)
    output[active] = output[active] * 0.92 + 255.0 * (gloss[active, None] * 0.05)
    return np.clip(output, 0, 255).astype(np.uint8)


def quad_rect_size(points: list[tuple[float, float]]) -> tuple[int, int]:
    tl, tr, br, bl = order_points(points)
    width = max(math.dist(tl, tr), math.dist(bl, br))
    height = max(math.dist(tl, bl), math.dist(tr, br))
    return max(2, int(round(width))), max(2, int(round(height)))


def expand_quad(points: list[tuple[float, float]], factor: float = 1.65) -> list[tuple[float, float]]:
    """扩大背景检测区域，但不改变前景实际贴图的四个控制点。"""
    ordered = np.asarray(order_points(points), dtype=np.float32)
    center = np.mean(ordered, axis=0, keepdims=True)
    expanded = center + (ordered - center) * float(max(1.0, factor))
    return [(float(x), float(y)) for x, y in expanded]


def estimate_screen_green(screen_rgb: np.ndarray, ignore_mask: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    green = (hue >= 35) & (hue <= 95) & (saturation > 45) & (value > 60)
    if ignore_mask is not None:
        green &= ignore_mask < 20
    if np.count_nonzero(green) < 24:
        return np.array([85, 205, 75], dtype=np.float32)
    return np.median(screen_rgb[green], axis=0).astype(np.float32)


def detect_white_cat_on_screen(screen_rgb: np.ndarray) -> np.ndarray:
    height, width = screen_rgb.shape[:2]
    if height < 8 or width < 8:
        return np.zeros((height, width), dtype=np.uint8)

    hsv = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    rgb = screen_rgb.astype(np.float32)
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    green = (hue >= 35) & (hue <= 95) & (saturation > 45) & (value > 55)
    bright_seed = (luma > 168) & (saturation < 105) & ~green
    pale_support = (luma > 126) & (saturation < 138) & ~green

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    seed = cv2.morphologyEx(bright_seed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    support = cv2.morphologyEx(pale_support.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)

    label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(support, connectivity=8)
    if label_count <= 1:
        return np.zeros((height, width), dtype=np.uint8)

    min_area = max(80, int(width * height * 0.012))
    seed_labels = labels[seed > 0]
    output = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, label_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        if np.count_nonzero(seed_labels == label) < max(20, area * 0.025):
            continue
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if comp_w > width * 0.72 and comp_h < height * 0.16:
            continue
        output[component] = 255

    if not np.any(output):
        return output

    output = fill_mask_holes(output)
    output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    output = cv2.dilate(output, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    # 白猫可以占据屏幕较大区域，不能只用面积过小的阈值误杀；
    # 真正的整屏误检通常会同时贴满四周边界。
    ys, xs = np.where(output > 0)
    if len(xs) > 0:
        box_width = int(xs.max() - xs.min() + 1)
        box_height = int(ys.max() - ys.min() + 1)
        area_ratio = np.count_nonzero(output) / float(height * width)
        touches_many_borders = (
            xs.min() <= width * 0.02
            and xs.max() >= width * 0.98
            and ys.min() <= height * 0.02
            and ys.max() >= height * 0.98
        )
        if area_ratio > 0.50 or (box_width > width * 0.92 and box_height > height * 0.92):
            return np.zeros((height, width), dtype=np.uint8)
        if touches_many_borders and area_ratio > 0.28:
            return np.zeros((height, width), dtype=np.uint8)
    return output.astype(np.uint8)


def replace_white_cat_with_green_screen(
    background: np.ndarray,
    points: list[tuple[float, float]],
) -> np.ndarray:
    if len(points) != 4:
        return background

    ordered = order_points(points)
    width, height = quad_rect_size(ordered)
    src = np.array(ordered, dtype=np.float32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    screen = warp_perspective(background, src, dst, (width, height), is_mask=False)
    cat_mask = detect_white_cat_on_screen(screen)
    if np.count_nonzero(cat_mask) < max(80, int(width * height * 0.008)):
        return background
    if np.count_nonzero(cat_mask) > int(width * height * 0.50):
        return background

    green = estimate_screen_green(screen, cat_mask)
    soft = feather_mask(cat_mask, max(5, int(round(min(width, height) * 0.018)))).astype(np.float32) / 255.0
    replaced_screen = screen.astype(np.float32) * (1.0 - soft[..., None]) + green[None, None, :] * soft[..., None]

    patched = warp_perspective(replaced_screen.astype(np.uint8), dst, src, (background.shape[1], background.shape[0]), is_mask=False)
    patched_mask = warp_soft_mask(cat_mask, dst, src, (background.shape[1], background.shape[0])).astype(np.float32) / 255.0
    patched_mask = feather_mask((patched_mask * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    result = background.astype(np.float32) * (1.0 - patched_mask[..., None]) + patched.astype(np.float32) * patched_mask[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def cylinder_body_box(
    points: list[tuple[float, float]],
    output_size: tuple[int, int],
    cover_ratio: float,
) -> tuple[int, int, int, int]:
    out_w, out_h = output_size
    ordered = order_points(points)
    xs = [point[0] for point in ordered]
    ys = [point[1] for point in ordered]
    x1 = float(np.clip(min(xs), 0, out_w - 1))
    x2 = float(np.clip(max(xs), 0, out_w - 1))
    y1 = float(np.clip(min(ys), 0, out_h - 1))
    y2 = float(np.clip(max(ys), 0, out_h - 1))

    safe_width = max(2.0, x2 - x1 + 1.0)
    center_x = (x1 + x2) / 2.0
    cover = float(np.clip(cover_ratio, 0.30, 1.20))
    target_width = safe_width * cover
    x1 = center_x - target_width / 2.0
    x2 = center_x + target_width / 2.0

    return (
        int(np.clip(round(x1), 0, out_w - 2)),
        int(np.clip(round(y1), 0, out_h - 2)),
        int(np.clip(round(x2), 1, out_w - 1)),
        int(np.clip(round(y2), 1, out_h - 1)),
    )


def cylinder_body_quad(
    points: list[tuple[float, float]],
    output_size: tuple[int, int],
    cover_ratio: float,
) -> np.ndarray:
    out_w, out_h = output_size
    quad = np.asarray(order_points(points), dtype=np.float32)
    quad[:, 0] = np.clip(quad[:, 0], 0, out_w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, out_h - 1)
    tl, tr, br, bl = quad

    def sample(u: float, v: float) -> np.ndarray:
        top = tl * (1.0 - u) + tr * u
        bottom = bl * (1.0 - u) + br * u
        return top * (1.0 - v) + bottom * v

    cover = float(np.clip(cover_ratio, 0.30, 1.20))
    u1 = 0.5 - cover / 2.0
    u2 = 0.5 + cover / 2.0
    v1, v2 = 0.0, 1.0

    body = np.asarray(
        [
            sample(u1, v1),
            sample(u2, v1),
            sample(u2, v2),
            sample(u1, v2),
        ],
        dtype=np.float32,
    )
    body[:, 0] = np.clip(body[:, 0], 0, out_w - 1)
    body[:, 1] = np.clip(body[:, 1], 0, out_h - 1)
    return body


def bounding_box_from_quad(quad: np.ndarray, output_size: tuple[int, int]) -> tuple[int, int, int, int]:
    out_w, out_h = output_size
    xs = quad[:, 0]
    ys = quad[:, 1]
    x1 = int(np.clip(math.floor(float(xs.min())), 0, out_w - 2))
    y1 = int(np.clip(math.floor(float(ys.min())), 0, out_h - 2))
    x2 = int(np.clip(math.ceil(float(xs.max())), x1 + 1, out_w - 1))
    y2 = int(np.clip(math.ceil(float(ys.max())), y1 + 1, out_h - 1))
    return x1, y1, x2, y2


def cylinder_surface_from_points(
    points: list[tuple[float, float]],
    output_size: tuple[int, int],
    cover_ratio: float,
    bend_strength: float,
    top_arc_factor: float = 1.0,
    bottom_arc_factor: float = 1.0,
) -> CylinderSurface:
    out_w, out_h = output_size
    tl, tr, br, bl = order_points(points)
    left_x = (tl[0] + bl[0]) / 2.0
    right_x = (tr[0] + br[0]) / 2.0
    if left_x > right_x:
        left_x, right_x = right_x, left_x
        tl, tr, br, bl = tr, tl, bl, br

    selected_width = max(2.0, right_x - left_x)
    center_x = (left_x + right_x) / 2.0
    cover = float(np.clip(cover_ratio, 0.30, 1.20))
    x1 = center_x - selected_width * cover / 2.0
    x2 = center_x + selected_width * cover / 2.0
    x1_i = int(np.clip(math.floor(x1), 0, out_w - 2))
    x2_i = int(np.clip(math.ceil(x2), x1_i + 1, out_w - 1))

    xs = np.arange(x1_i, x2_i + 1, dtype=np.float32)
    full_t = (xs - left_x) / max(1.0, right_x - left_x)
    top_line = tl[1] * (1.0 - full_t) + tr[1] * full_t
    bottom_line = bl[1] * (1.0 - full_t) + br[1] * full_t
    avg_height = max(8.0, float(np.mean(bottom_line - top_line)))
    bend = float(np.clip(bend_strength, 0.0, 1.0))
    clipped_t = np.clip(full_t, 0.0, 1.0)
    top_phase = np.sin(np.pi * clipped_t)
    side_distance = np.abs(clipped_t * 2.0 - 1.0)
    bottom_phase = np.clip(1.0 - side_distance**2.0, 0.0, 1.0)
    arc_depth = min(avg_height * 0.16, selected_width * 0.13) * (0.55 + 0.65 * bend)

    top_factor = float(np.clip(top_arc_factor, 0.0, 2.2))
    bottom_factor = float(np.clip(bottom_arc_factor, 0.0, 2.5))
    top = top_line - arc_depth * 0.72 * top_factor * top_phase
    bottom = bottom_line + arc_depth * 1.05 * bottom_factor * bottom_phase
    min_gap = max(8.0, avg_height * 0.35)
    tight = bottom - top < min_gap
    if np.any(tight):
        center = (top[tight] + bottom[tight]) / 2.0
        top[tight] = center - min_gap / 2.0
        bottom[tight] = center + min_gap / 2.0

    y1_i = int(np.clip(math.floor(float(np.min(top))), 0, out_h - 2))
    y2_i = int(np.clip(math.ceil(float(np.max(bottom))), y1_i + 1, out_h - 1))
    return CylinderSurface(
        (x1_i, y1_i, x2_i, y2_i),
        (top - y1_i).astype(np.float32),
        (bottom - y1_i).astype(np.float32),
        top_factor,
        bottom_factor,
    )


def cylinder_surface_polygon(surface: CylinderSurface, step: int = 3) -> list[tuple[float, float]]:
    x1, y1, _x2, _y2 = surface.box
    width = len(surface.top)
    indices = list(range(0, width, max(1, step)))
    if indices[-1] != width - 1:
        indices.append(width - 1)
    top = [(x1 + i, y1 + float(surface.top[i])) for i in indices]
    bottom = [(x1 + i, y1 + float(surface.bottom[i])) for i in reversed(indices)]
    return top + bottom


def cylinder_instance_quads(
    points: list[tuple[float, float]],
    output_size: tuple[int, int],
) -> list[list[tuple[float, float]]]:
    """Build top/middle/bottom target quads from the user-selected bottom quad."""
    out_w, out_h = output_size
    base = np.asarray(order_points(points), dtype=np.float32)
    base_center = np.mean(base, axis=0)
    top_mid = (base[0] + base[1]) * 0.5
    bottom_mid = (base[3] + base[2]) * 0.5
    down_axis = bottom_mid - top_mid
    height = float(np.linalg.norm(down_axis))
    if height <= 1.0:
        return [
            [(float(x), float(y)) for x, y in base],
            [(float(x), float(y)) for x, y in base],
            [(float(x), float(y)) for x, y in base],
        ]

    down_axis /= height
    gap_ratio = 0.70

    def build_layout(upper_scale: float) -> list[list[tuple[float, float]]]:
        middle_scale = 0.75 * upper_scale
        top_scale = 0.50 * upper_scale
        centers = [base_center.copy()]
        current_center = base_center.copy()
        current_scale = 1.0
        for next_scale in (middle_scale, top_scale):
            lift = ((current_scale + next_scale) * 0.5 + gap_ratio * upper_scale) * height
            current_center = current_center - down_axis * lift
            centers.append(current_center.copy())
            current_scale = next_scale

        quads: list[list[tuple[float, float]]] = []
        for scale, center in zip((top_scale, middle_scale, 1.0), reversed(centers)):
            scaled = center + (base - base_center) * scale
            quads.append([(float(x), float(y)) for x, y in scaled])
        return quads

    def fits_inside(quad_list: list[list[tuple[float, float]]]) -> bool:
        for quad in quad_list[:2]:
            coords = np.asarray(quad, dtype=np.float32)
            if (
                np.any(coords[:, 0] < 1.0)
                or np.any(coords[:, 0] > out_w - 2.0)
                or np.any(coords[:, 1] < 1.0)
                or np.any(coords[:, 1] > out_h - 2.0)
            ):
                return False
        return True

    upper_scale = 1.0
    if not fits_inside(build_layout(upper_scale)):
        low, high = 0.05, 1.0
        for _ in range(28):
            candidate = (low + high) * 0.5
            if fits_inside(build_layout(candidate)):
                low = candidate
            else:
                high = candidate
        upper_scale = low
    return build_layout(upper_scale)


def quad_mask(shape: tuple[int, int], quad: np.ndarray, feather: int = 0) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.round(quad).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillConvexPoly(mask, polygon, 255)
    return feather_mask(mask, feather) if feather > 0 else mask


def robust_normalize(values: np.ndarray, active: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    if not np.any(active):
        return np.zeros_like(values, dtype=np.float32)
    scale = float(np.percentile(np.abs(values[active]), percentile))
    if scale <= 1e-3:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, -1.0, 1.0).astype(np.float32)


def apply_displacement_map(
    foreground: np.ndarray,
    mask: np.ndarray,
    background_patch: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = mask > 8
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.01 or not np.any(active):
        return foreground, mask

    bg = background_patch.astype(np.float32)
    bg_luma = bg @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sigma = max(3.0, min(mask.shape) / 22.0)
    smooth = cv2.GaussianBlur(bg_luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = bg_luma - smooth
    detail_norm = robust_normalize(detail, active, percentile=92.0)
    grad_x = robust_normalize(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3), active)
    grad_y = robust_normalize(cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3), active)

    height, width = mask.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    influence = feather_mask(mask, 6).astype(np.float32) / 255.0
    amount = 0.45 + strength * 4.2
    dx = (detail_norm * 0.55 + grad_x * 0.45) * amount * influence
    dy = (detail_norm * 0.18 + grad_y * 0.35) * amount * influence
    map_x = np.clip(xx + dx, 0, width - 1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, height - 1).astype(np.float32)

    displaced_rgb = cv2.remap(
        foreground,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    displaced_mask = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return displaced_rgb.astype(np.uint8), displaced_mask.astype(np.uint8)


def adapt_foreground_to_cylinder_light(
    foreground: np.ndarray,
    background_patch: np.ndarray,
    mask: np.ndarray,
    theta: np.ndarray,
    bend_strength: float,
    light_strength: float,
) -> np.ndarray:
    light_strength = float(np.clip(light_strength, 0.0, 1.0))
    if light_strength <= 0.01:
        return foreground
    active = mask > 8
    if not np.any(active):
        return foreground

    fg = foreground.astype(np.float32)
    bg = background_patch.astype(np.float32)
    fg_luma = fg @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    bg_luma = bg @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    # 光影适配保持柔和，避免把柱面背景的亮部复制成前景高光。
    strength = float(np.clip(light_strength * 0.90, 0.0, 1.0))

    fg_mean = float(np.mean(fg_luma[active]))
    bg_mean = float(np.mean(bg_luma[active]))
    gain = float(np.clip((bg_mean + 18.0) / (fg_mean + 18.0), 0.58, 1.03))

    smooth_sigma = max(7.0, min(mask.shape) / 10.0)
    smooth_light = cv2.GaussianBlur(bg_luma, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
    smooth_mean = float(np.mean(smooth_light[active]))
    surface_shading = np.clip(smooth_light / max(smooth_mean, 1.0), 0.80, 1.16)
    surface_shading = 1.0 + (surface_shading - 1.0) * (0.65 + 0.25 * strength)

    texture_base = cv2.GaussianBlur(bg_luma, (0, 0), sigmaX=3.2, sigmaY=3.2)
    texture_gain = np.clip((bg_luma + 12.0) / (texture_base + 12.0), 0.94, 1.06)
    texture_gain = 1.0 + (texture_gain - 1.0) * (0.22 + 0.28 * strength)

    cylinder_light = cylinder_shading_weight(theta, bend_strength, light_strength)

    contrast = 0.92 + 0.07 * np.clip(bg_mean / 180.0, 0.0, 1.0)
    adjusted = (fg - 128.0) * contrast + 128.0
    adjusted *= gain * surface_shading[..., None] * texture_gain[..., None] * cylinder_light[..., None]
    adjusted = harmonize_cylinder_tone(np.clip(adjusted, 0, 255).astype(np.uint8), background_patch, mask, strength)
    adjusted = adjusted.astype(np.float32) * (1.0 - 0.10 * strength) + bg * (0.10 * strength)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def harmonize_cylinder_tone(
    foreground: np.ndarray,
    background_patch: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    active = mask > 8
    if not np.any(active):
        return foreground

    amount = float(np.clip(0.10 + 0.25 * strength, 0.0, 0.36))
    fg = foreground.astype(np.float32)
    bg = background_patch.astype(np.float32)

    fg_active = fg[active]
    bg_active = bg[active]
    fg_mean = np.mean(fg_active, axis=0)
    bg_mean = np.mean(bg_active, axis=0)
    target_mean = fg_mean * (1.0 - amount) + bg_mean * amount
    channel_gain = np.clip((target_mean + 18.0) / (fg_mean + 18.0), 0.58, 1.05)
    toned = fg * channel_gain

    toned = 255.0 * np.power(np.clip(toned / 255.0, 0.0, 1.0), 1.03 + 0.05 * strength)
    hsv = cv2.cvtColor(np.clip(toned, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    # 柱面材质采用偏灰的哑光色调，保留颜色信息但降低鲜艳度。
    hsv[:, :, 1] *= 0.72 - 0.12 * strength
    toned = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    material = (0.025 + 0.05 * strength) * (feather_mask(mask, 5).astype(np.float32) / 255.0)
    toned = toned * (1.0 - material[..., None]) + bg * material[..., None]
    return np.clip(toned, 0, 255).astype(np.uint8)


def cylinder_shading_weight(theta: np.ndarray, bend_strength: float, light_strength: float) -> np.ndarray:
    half_theta = max(float(np.max(np.abs(theta))), 1e-3)
    side_amount = np.clip(np.abs(theta) / half_theta, 0.0, 1.0)
    cos_weight = np.clip(np.cos(theta), 0.42, 1.0)
    bend = float(np.clip(bend_strength, 0.0, 1.0))
    light = float(np.clip(light_strength, 0.0, 1.0))
    side_shadow = 1.0 - (0.24 + 0.28 * bend) * (side_amount**1.75)
    center_highlight = 1.0 + (0.012 + 0.018 * light) * (1.0 - side_amount**2.6)
    weight = (0.62 + 0.38 * cos_weight) * side_shadow * center_highlight
    return np.clip(weight, 0.42, 1.04).astype(np.float32)


def cylinder_visible_alpha(theta: np.ndarray, bend_strength: float) -> np.ndarray:
    half_theta = max(float(np.max(np.abs(theta))), 1e-3)
    side_amount = np.clip(np.abs(theta) / half_theta, 0.0, 1.0)
    bend = float(np.clip(bend_strength, 0.0, 1.0))
    rolloff = 1.0 - (0.12 + 0.16 * bend) * (side_amount**3.0)
    edge_softness = np.clip((1.0 - side_amount) / 0.10, 0.0, 1.0)
    return np.clip(rolloff * edge_softness, 0.0, 1.0).astype(np.float32)


def add_cylinder_vertical_depth(mask: np.ndarray, theta: np.ndarray, bend_strength: float) -> np.ndarray:
    height, width = mask.shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_shadow = np.exp(-((y - 0.03) / 0.055) ** 2)
    bottom_shadow = np.exp(-((y - 0.97) / 0.060) ** 2)
    side = np.clip(np.abs(theta) / max(float(np.max(np.abs(theta))), 1e-3), 0.0, 1.0)
    depth = 1.0 - (0.10 + 0.08 * bend_strength) * (top_shadow + bottom_shadow)
    depth = depth * (1.0 - 0.055 * side**2)
    return np.clip(mask.astype(np.float32) * depth, 0, 255).astype(np.uint8)


def imprint_cylinder_texture(
    foreground: np.ndarray,
    background_patch: np.ndarray,
    mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    amount = float(np.clip(strength, 0.0, 1.0))
    if amount <= 0.01:
        return foreground
    active = mask > 8
    if not np.any(active):
        return foreground

    fg = foreground.astype(np.float32)
    bg = background_patch.astype(np.float32)
    bg_luma = bg @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    broad = cv2.GaussianBlur(bg_luma, (0, 0), sigmaX=10.0, sigmaY=10.0)
    fine = cv2.GaussianBlur(bg_luma, (0, 0), sigmaX=2.0, sigmaY=2.0) - broad
    fine_norm = robust_normalize(fine, active, percentile=90.0)
    relief = cv2.Laplacian(cv2.GaussianBlur(bg_luma, (0, 0), sigmaX=1.4, sigmaY=1.4), cv2.CV_32F)
    relief_norm = robust_normalize(relief, active, percentile=92.0)
    texture_gain = 1.0 + fine_norm * (0.055 + 0.11 * amount) + relief_norm * (0.018 + 0.035 * amount)

    material_mix = (0.035 + 0.075 * amount) * (feather_mask(mask, 5).astype(np.float32) / 255.0)
    textured = fg * texture_gain[..., None]
    textured = textured * (1.0 - material_mix[..., None]) + bg * material_mix[..., None]
    return np.clip(textured, 0, 255).astype(np.uint8)


def cylinder_warp(
    foreground: np.ndarray,
    mask: np.ndarray,
    surface: CylinderSurface,
    output_size: tuple[int, int],
    bend_strength: float,
    light_strength: float,
    texture_strength: float,
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    out_w, out_h = output_size
    x1, y1, x2, y2 = surface.box
    x1, x2 = sorted((max(0, x1), min(out_w - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(out_h - 1, y2)))
    width = max(2, x2 - x1 + 1)
    height = max(2, y2 - y1 + 1)

    fg_h, fg_w = foreground.shape[:2]
    bend = float(np.clip(bend_strength, 0.0, 1.0))
    light = float(np.clip(light_strength, 0.0, 1.0))
    texture = float(np.clip(texture_strength, 0.0, 1.0))
    theta_max = math.radians(70.0 + bend * 95.0)
    theta_half = theta_max / 2.0
    sin_half = max(0.2, math.sin(theta_half))

    local_x, local_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    # 反向映射：目标柱面 patch 上的每个像素反推前景图采样坐标，避免正向变形产生空洞。
    norm_x = (local_x / max(1, width - 1) * 2.0 - 1.0) * sin_half
    theta = np.arcsin(np.clip(norm_x, -0.999, 0.999)).astype(np.float32)
    u = theta / theta_max + 0.5

    top = surface.top[:width][None, :]
    bottom = surface.bottom[:width][None, :]
    column_height = np.maximum(1.0, bottom - top)
    raw_v = (local_y - top) / column_height
    inside_surface = (local_y >= top) & (local_y <= bottom)
    side_amount = np.clip(np.abs(theta) / max(theta_half, 1e-3), 0.0, 1.0)
    row_phase = np.sin(np.pi * np.clip(raw_v, 0.0, 1.0))
    row_bow = (0.016 + 0.034 * bend) * (1.0 - side_amount**2.0) * row_phase
    v = raw_v - row_bow
    vertical_bend = (np.cos(theta) - math.cos(theta_half)) / max(1e-3, 1.0 - math.cos(theta_half))
    v = np.clip(v + (vertical_bend - 0.5) * 0.012 * bend, 0.0, 1.0)

    map_x = (u * (fg_w - 1)).astype(np.float32)
    map_y = (v * (fg_h - 1)).astype(np.float32)
    sampled = cv2.remap(
        foreground,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    sampled_mask = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    top_edge_feather = max(2.0, 2.5 + bend * 3.5)
    bottom_edge_feather = max(6.0, 7.0 + bend * 8.0)
    top_alpha = np.clip((local_y - top) / top_edge_feather, 0.0, 1.0)
    bottom_alpha = np.clip((bottom - local_y) / bottom_edge_feather, 0.0, 1.0)
    surface_alpha = np.minimum(
        top_alpha * top_alpha * (3.0 - 2.0 * top_alpha),
        bottom_alpha * bottom_alpha * (3.0 - 2.0 * bottom_alpha),
    )
    sampled_mask = (sampled_mask.astype(np.float32) * surface_alpha).astype(np.uint8)
    sampled_mask[~inside_surface] = 0
    sampled_mask = feather_mask(sampled_mask, max(2, int(round(2 + bend * 3))))
    sampled_mask = np.clip(
        sampled_mask.astype(np.float32) * cylinder_visible_alpha(theta, bend),
        0,
        255,
    ).astype(np.uint8)
    sampled_mask = add_cylinder_vertical_depth(sampled_mask, theta, bend)

    if background is not None:
        background_patch = background[y1 : y2 + 1, x1 : x2 + 1]
        sampled, sampled_mask = apply_displacement_map(sampled, sampled_mask, background_patch, light * 2.0)
        sampled = adapt_foreground_to_cylinder_light(sampled, background_patch, sampled_mask, theta, bend, light)
        sampled = imprint_cylinder_texture(sampled, background_patch, sampled_mask, texture)

    warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    warped_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    warped[y1 : y2 + 1, x1 : x2 + 1] = sampled
    warped_mask[y1 : y2 + 1, x1 : x2 + 1] = sampled_mask
    return warped, warped_mask


class PasteMappingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("平面/柱面贴图系统")
        self.root.geometry("1698x944")
        self.root.minsize(1280, 760)
        self.root.configure(bg=BG)

        self.background_files = list_images(BACKGROUND_DIR)
        self.foreground_files = list_images(FOREGROUND_DIR)
        self.background_path: Path | None = None
        self.foreground_path: Path | None = None
        self.background_rgb: np.ndarray | None = None
        self.green_background_rgb: np.ndarray | None = None
        self.foreground_rgb: np.ndarray | None = None
        self.foreground_mask: np.ndarray | None = None
        self.result_rgb: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.mapped_rgb: np.ndarray | None = None
        self.mapped_mask: np.ndarray | None = None
        self.transformed_preview_rgb: np.ndarray | None = None
        self.transformed_preview_mask: np.ndarray | None = None
        self.brush_map: np.ndarray | None = None
        self.points: list[tuple[float, float]] = []
        self.undo_stack: list[UndoState] = []
        self.display_geometry = DisplayGeometry()
        self.canvas_photo: ImageTk.PhotoImage | None = None
        self.foreground_preview_photo: ImageTk.PhotoImage | None = None
        self.mapped_preview_photo: ImageTk.PhotoImage | None = None
        self.load_time: float | None = None
        self.interaction_count = 0
        self.drag_index: int | None = None
        self.drag_mode: str | None = None
        self.drag_anchor: tuple[float, float] | None = None
        self.drag_original_points: list[tuple[float, float]] = []
        self.drag_started = False
        self.cylinder_top_arc_factor = 1.0
        self.cylinder_bottom_arc_factor = 1.68
        self.mode_text_var = tk.StringVar(value="当前模式：平面贴图")

        self.mode_var = tk.StringVar(value="laptop")
        self.background_var = tk.StringVar()
        self.foreground_var = tk.StringVar()
        self.blend_var = tk.StringVar(value="梯度融合")
        self.opacity_var = tk.IntVar(value=95)
        self.feather_var = tk.IntVar(value=6)
        self.brightness_adjust_var = tk.IntVar(value=82)
        self.hue_shift_var = tk.IntVar(value=0)
        self.temperature_var = tk.IntVar(value=0)
        self.saturation_var = tk.IntVar(value=100)
        self.contrast_var = tk.IntVar(value=100)
        self.brush_size_var = tk.IntVar(value=42)
        self.brush_strength_var = tk.IntVar(value=72)
        self.brush_enabled = False
        self.brush_stroking = False
        self.brightness_var = tk.BooleanVar(value=True)
        self.wrap_var = tk.IntVar(value=86)
        self.light_var = tk.IntVar(value=35)
        self.cover_var = tk.IntVar(value=100)
        self.cylinder_angle_var = tk.IntVar(value=125)
        self.texture_var = tk.IntVar(value=5)
        self.time_var = tk.StringVar(value="操作时间 0.0 s")
        self.count_var = tk.StringVar(value="交互次数 0")
        self.status_var = tk.StringVar(value="读取背景图后开始计时")

        self._build_ui()
        self._load_defaults()
        self.root.after(200, self._tick)

    def _build_ui(self) -> None:
        self._configure_style()
        shell = tk.Frame(self.root, bg=SOFT_BLUE, highlightthickness=0, bd=0)
        shell.pack(fill=tk.BOTH, expand=True, padx=36, pady=18)

        self._build_topbar(shell)

        body = tk.Frame(shell, bg=SOFT_BLUE)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=(10, 0))

        left = tk.Frame(body, bg=SOFT_BLUE, width=292)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        left.pack_propagate(False)
        tk.Label(left, text="编辑操作", bg=SOFT_BLUE, fg=INK, font=(UI_FONT, 12, "bold")).pack(anchor="w", pady=(0, 6))
        self._build_parameter_controls(left)
        self._build_log(left)

        result_panel = tk.Frame(body, bg=PANEL, highlightthickness=1, highlightbackground="#d8e8fb")
        result_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas_header = tk.Frame(result_panel, bg=PANEL)
        canvas_header.pack(fill=tk.X, padx=18, pady=(16, 8))
        tk.Label(canvas_header, text="结果图像界面", bg=PANEL, fg=INK, font=(UI_FONT, 12, "bold")).pack(side=tk.LEFT)

        self.canvas_wrap = tk.Frame(result_panel, bg="#eef4fb")
        self.canvas_wrap.pack(fill=tk.X, padx=18, pady=(0, 10))
        self.source_canvas = tk.Canvas(
            self.canvas_wrap,
            bg="#eef4fb",
            width=CANVAS_TARGET_WIDTH,
            height=CANVAS_TARGET_HEIGHT,
            highlightthickness=0,
        )
        self.source_canvas.pack(padx=0, pady=0)
        self._build_canvas_overlays(self.canvas_wrap)

        bottom_tools = tk.Frame(result_panel, bg=PANEL)
        bottom_tools.pack(fill=tk.X, padx=18, pady=(0, 14))
        tk.Label(bottom_tools, text="视图模式", bg=PANEL, fg=MUTED, font=(UI_FONT, 9, "bold")).pack(side=tk.LEFT)
        tk.Label(bottom_tools, text="单张图", bg="#ffffff", fg="#49627f", font=(UI_FONT, 9), padx=20, pady=8).pack(side=tk.LEFT, padx=(14, 0))
        tk.Label(bottom_tools, text="单图编辑", bg="#eaf3ff", fg=BLUE, font=(UI_FONT, 9, "bold"), padx=20, pady=8).pack(side=tk.LEFT)
        tk.Label(bottom_tools, text="参考线", bg="#ffffff", fg="#8aa0bd", font=(UI_FONT, 9), padx=20, pady=8).pack(side=tk.RIGHT, padx=(8, 0))

        preview_column = tk.Frame(body, bg=SOFT_BLUE, width=330)
        preview_column.pack(side=tk.LEFT, fill=tk.Y, padx=(14, 0))
        preview_column.pack_propagate(False)
        self.foreground_preview_frame = self._build_preview_panel(preview_column, "插入目标图像")
        self.foreground_preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 7))
        self.mapped_preview_frame = self._build_preview_panel(preview_column, "几何变换后的目标图像")
        self.mapped_preview_frame.pack(fill=tk.BOTH, expand=True, pady=(7, 0))

        self._build_statusbar(shell)

        self.source_canvas.bind("<Configure>", lambda _event: self.redraw_source())
        self.source_canvas.bind("<Button-1>", self._on_canvas_press)
        self.source_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.source_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.source_canvas.bind("<Double-Button-1>", self._on_canvas_double_click)

    def _build_preview_panel(self, parent: tk.Frame, title: str) -> tk.Frame:
        panel = tk.Frame(parent, bg="#f7fbff", highlightthickness=1, highlightbackground="#d8e8fb")
        tk.Label(panel, text=title, bg="#f7fbff", fg="#29476b", font=(UI_FONT, 9, "bold")).pack(
            anchor="w", padx=10, pady=(5, 3)
        )
        canvas = tk.Canvas(
            panel,
            bg="#eef4fb",
            height=PREVIEW_CANVAS_HEIGHT,
            highlightthickness=0,
        )
        canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        if title == "插入目标图像":
            self.foreground_preview_canvas = canvas
        else:
            self.mapped_preview_canvas = canvas
        return panel

    def _build_canvas_overlays(self, parent: tk.Frame) -> None:
        palette = tk.Frame(parent, bg="#ffffff", highlightthickness=1, highlightbackground="#d8e8fb")
        palette.place(relx=0.018, rely=0.18)
        for text, command in (
            ("↖", lambda: self.status_var.set("拖动贴图主体可移动 mask")),
            ("✋", lambda: self.status_var.set("按住 mask 内部并拖动即可平移")),
        ):
            tk.Button(
                palette,
                text=text,
                command=command,
                bg="#ffffff",
                fg="#29476b",
                activebackground="#eef5ff",
                relief=tk.FLAT,
                bd=0,
                width=3,
                height=1,
                font=(UI_FONT, 12),
                cursor="hand2",
            ).pack(padx=5, pady=5)

    def _build_topbar(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg=SOFT_BLUE)
        top.pack(fill=tk.X, padx=18, pady=(16, 4))

        logo = tk.Label(top, text="⌕", bg="#5b8ff9", fg="#ffffff", font=(UI_FONT, 20, "bold"), width=2, height=1)
        logo.pack(side=tk.LEFT)
        title_box = tk.Frame(top, bg=SOFT_BLUE)
        title_box.pack(side=tk.LEFT, padx=(12, 26))
        tk.Label(title_box, text="平面 / 柱面贴图系统", bg=SOFT_BLUE, fg=INK, font=(UI_FONT, 15, "bold")).pack(anchor="w")
        tk.Label(title_box, text="Image Mapping Studio", bg=SOFT_BLUE, fg="#41618f", font=(UI_FONT, 10)).pack(anchor="w")

        self._toolbar_button(top, "▣  打开背景图", self.open_background).pack(side=tk.LEFT, padx=7)
        self._toolbar_button(top, "▧  打开前景图", self.open_foreground).pack(side=tk.LEFT, padx=7)
        self.laptop_button = self._toolbar_button(top, "⊞  平面贴图", lambda: self.set_mode("laptop"))
        self.laptop_button.pack(side=tk.LEFT, padx=6)
        self.cylinder_button = self._toolbar_button(top, "▤  柱面贴图", lambda: self.set_mode("cylinder"))
        self.cylinder_button.pack(side=tk.LEFT, padx=6)

        right = tk.Frame(top, bg=SOFT_BLUE)
        right.pack(side=tk.RIGHT)
        self._toolbar_button(right, "◉  融合图像", self.fuse_image, primary=True).pack(side=tk.LEFT, padx=7)
        self._toolbar_button(right, "↶  撤销", self.undo).pack(side=tk.LEFT, padx=7)
        self._toolbar_button(right, "↻  重置", self.reset_result).pack(side=tk.LEFT, padx=7)
        self._toolbar_button(right, "⇩  保存结果", self.save_result).pack(side=tk.LEFT, padx=7)
        self.refresh_mode_buttons()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=(UI_FONT, 9))
        style.configure("Root.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL, relief=tk.SOLID, borderwidth=1)
        style.configure("ImagePanel.TFrame", background=PANEL, relief=tk.SOLID, borderwidth=1)
        style.configure("Title.TLabel", background=PANEL, foreground=INK, font=(UI_FONT, 15, "bold"))
        style.configure("PaneTitle.TLabel", background=PANEL, foreground=INK, font=(UI_FONT, 10, "bold"))
        style.configure("Field.TLabel", background=PANEL, foreground=MUTED, font=(UI_FONT, 9, "bold"))
        style.configure("Status.TLabel", background=BG, foreground=MUTED)
        style.configure("Badge.TLabel", background="#e9efff", foreground="#1d4ed8", padding=(10, 6), font=(UI_FONT, 9, "bold"))
        style.configure("TButton", padding=(8, 6), font=(UI_FONT, 9))
        style.configure("Primary.TButton", padding=(8, 7), font=(UI_FONT, 9, "bold"))
        style.configure("TRadiobutton", background=PANEL, foreground=INK)
        style.configure("TCheckbutton", background=PANEL, foreground=INK)
        style.configure(
            "Clean.TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground="#29476b",
            borderwidth=0,
            relief=tk.FLAT,
            padding=(8, 4),
        )
        style.map("Clean.TCombobox", fieldbackground=[("readonly", "#ffffff")], selectbackground=[("readonly", "#ffffff")])

    def _build_parameter_controls(self, parent: tk.Frame) -> None:
        metrics = self._card(parent)
        self._card_title(metrics, "操作统计")
        tk.Label(metrics, textvariable=self.time_var, bg=CARD_BG, fg="#395b8f", font=(UI_FONT, 11, "bold")).pack(
            anchor="w", padx=14, pady=(0, 4)
        )
        tk.Label(metrics, textvariable=self.count_var, bg=CARD_BG, fg="#395b8f", font=(UI_FONT, 11, "bold")).pack(
            anchor="w", padx=14, pady=(0, 8)
        )

        mode_card = self._card(parent)
        self._card_title(mode_card, "模式参数")
        self._scale(mode_card, "弯曲强度", self.wrap_var, 0, 100, suffix="%")
        self._scale(mode_card, "水平覆盖", self.cover_var, 30, 120, suffix="%")
        self._scale(mode_card, "光影适配", self.light_var, 0, 100, suffix="%")
        self._scale(mode_card, "纹理强度", self.texture_var, 0, 50, suffix="%")

        blend_card = self._card(parent, border=False)
        self._card_title(blend_card, "融合效果")
        self._scale(blend_card, "透明度", self.opacity_var, 20, 100, suffix="%")
        self._scale(blend_card, "羽化强度", self.feather_var, 0, 20, suffix=" px")
        self._scale(blend_card, "前景亮度", self.brightness_adjust_var, 50, 150, suffix="%")
        self._field_label(blend_card, "融合模式").pack(fill=tk.X, padx=14, pady=(4, 1))
        blend = ttk.Combobox(
            blend_card,
            textvariable=self.blend_var,
            values=["普通融合", "羽化融合", "梯度融合", "Poisson融合"],
            state="readonly",
            style="Clean.TCombobox",
        )
        blend.pack(fill=tk.X, padx=14)
        blend.bind("<<ComboboxSelected>>", lambda _event: self.parameter_changed("切换融合方式"))
        self._switch_row(blend_card, "亮度匹配", self.brightness_var, lambda: self.parameter_changed("切换亮度匹配"))
        self._scale(blend_card, "笔刷大小", self.brush_size_var, 12, 120, suffix=" px", compact=True)
        self._scale(blend_card, "混合强度", self.brush_strength_var, 10, 100, suffix="%", compact=True)
        self.brush_button = self._small_button(blend_card, "启用混合笔刷", self.toggle_blend_brush)
        self.brush_button.pack(fill=tk.X, padx=14, pady=(2, 6))

        color_card = self._card(parent)
        self._card_title(color_card, "前景调色")
        self._scale(color_card, "色彩强度", self.saturation_var, 0, 200, suffix="%", compact=True)

        action_card = self._card(parent)
        self._card_title(action_card, "输出")
        self._small_button(action_card, "预览贴图", lambda: self.apply_mapping(count_interaction=False)).pack(
            fill=tk.X, padx=14, pady=(0, 5)
        )
        self._small_button(action_card, "清空控制点", self.clear_points).pack(fill=tk.X, padx=14, pady=(0, 5))
        self._small_button(action_card, "保存GUI截图", self.save_gui_snapshot).pack(fill=tk.X, padx=14, pady=(0, 8))

    def _build_action_controls(self, parent: tk.Frame) -> None:
        return

    def _build_log(self, parent: tk.Frame) -> None:
        card = self._card(parent, fill=tk.BOTH, expand=True)
        self._card_title(card, "交互日志")
        self.log_box = tk.Text(card, height=4, bg=CARD_BG, fg=INK, relief=tk.FLAT, wrap=tk.WORD, font=(UI_FONT, 9))
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))
        self.log_box.configure(state=tk.DISABLED)

    def _build_statusbar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground="#d8e8fb")
        bar.pack(fill=tk.X, padx=18, pady=(12, 16))
        tk.Label(bar, text="●", bg=PANEL, fg=GREEN, font=(UI_FONT, 13, "bold")).pack(side=tk.LEFT, padx=(18, 6), pady=11)
        tk.Label(bar, textvariable=self.status_var, bg=PANEL, fg="#284267", font=(UI_FONT, 10, "bold")).pack(side=tk.LEFT, padx=(0, 40), pady=11)
        tk.Label(bar, textvariable=self.time_var, bg=PANEL, fg="#395b8f", font=(UI_FONT, 10)).pack(side=tk.LEFT, padx=80)
        tk.Label(bar, textvariable=self.count_var, bg=PANEL, fg="#395b8f", font=(UI_FONT, 10)).pack(side=tk.LEFT, padx=40)
        tk.Label(bar, textvariable=self.mode_text_var, bg=PANEL, fg="#395b8f", font=(UI_FONT, 10)).pack(side=tk.LEFT, padx=40)

    def _card(self, parent: tk.Frame, fill: str = tk.X, expand: bool = False, border: bool = True) -> tk.Frame:
        card = tk.Frame(parent, bg=CARD_BG, highlightthickness=1 if border else 0, highlightbackground="#dce8f6", bd=0)
        card.pack(fill=fill, expand=expand, pady=(0, 6))
        return card

    def _card_title(self, parent: tk.Frame, text: str) -> None:
        tk.Label(parent, text=text, bg=CARD_BG, fg="#1d477f", font=(UI_FONT, 11, "bold")).pack(anchor="w", padx=14, pady=(8, 3))

    def _field_label(self, parent: tk.Frame, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg=CARD_BG, fg=MUTED, anchor="w", font=(UI_FONT, 9, "bold"))

    def _scale(
        self,
        parent: tk.Frame,
        title: str,
        variable: tk.IntVar,
        from_: int,
        to: int,
        suffix: str = "",
        compact: bool = False,
    ) -> None:
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill=tk.X, padx=14, pady=(1 if compact else 3, 0))
        tk.Label(row, text=title, bg=CARD_BG, fg="#49627f", font=(UI_FONT, 9)).pack(side=tk.LEFT)
        value_var = tk.StringVar(value=f"{variable.get()}{suffix}")
        tk.Label(row, textvariable=value_var, bg="#f0f5fb", fg="#49627f", font=(UI_FONT, 9), padx=8, pady=1).pack(side=tk.RIGHT)

        def update_value(_value: str, name: str = title) -> None:
            value_var.set(f"{variable.get()}{suffix}")
            self.parameter_changed(f"调整{name}", preview_only=True)

        scale = tk.Scale(
            parent,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=update_value,
            bg=CARD_BG,
            fg=INK,
            troughcolor="#e5e7eb",
            activebackground=BLUE,
            highlightthickness=0,
            relief=tk.FLAT,
            showvalue=False,
            sliderlength=12 if compact else 14,
            width=6 if compact else 8,
            bd=0,
        )
        scale.pack(fill=tk.X, padx=12, pady=(0, 0))
        scale.bind("<ButtonRelease-1>", lambda _event, name=title: self.finish_parameter_edit(f"调整{name}"))

    def _switch_row(self, parent: tk.Frame, title: str, variable: tk.BooleanVar, command) -> None:
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill=tk.X, padx=14, pady=(3, 6))
        tk.Label(row, text=title, bg=CARD_BG, fg="#49627f", font=(UI_FONT, 9)).pack(side=tk.LEFT)
        ttk.Checkbutton(row, variable=variable, command=command).pack(side=tk.RIGHT)

    def _toolbar_button(self, parent: tk.Frame, text: str, command, primary: bool = False) -> tk.Button:
        bg = ACTIVE_BLUE if primary else BUTTON_BG
        fg = "#ffffff" if primary else "#29476b"
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground="#1f6fd4" if primary else "#eef5ff",
            activeforeground=fg,
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=10,
            font=(UI_FONT, 10, "bold" if primary else "normal"),
            cursor="hand2",
        )

    def _small_button(self, parent: tk.Frame, text: str, command, primary: bool = False) -> tk.Button:
        return self._toolbar_button(parent, text, command, primary=primary)

    def _load_defaults(self) -> None:
        if self.background_files:
            preferred = self._find_named(self.background_files, "laptop-1") or self._find_named(self.background_files, "laptop") or self.background_files[0]
            self.background_var.set(preferred.name)
            self.load_background(preferred)
        if self.foreground_files:
            preferred = self._find_named(self.foreground_files, "LENA") or self.foreground_files[0]
            self.foreground_var.set(preferred.name)
            self.load_foreground(preferred)
        self.status_var.set("请在背景图上人工点击 4 个角点选择贴图位置")

    def _find_named(self, files: list[Path], name: str) -> Path | None:
        name = name.lower()
        for path in files:
            if name in path.name.lower():
                return path
        return None

    def load_selected_background(self) -> None:
        match = self._find_exact(self.background_files, self.background_var.get())
        if match:
            self.load_background(match)
            self.log_action("选择背景图")

    def load_selected_foreground(self) -> None:
        match = self._find_exact(self.foreground_files, self.foreground_var.get())
        if match:
            self.load_foreground(match)
            self.log_action("选择前景图")

    def _find_exact(self, files: list[Path], filename: str) -> Path | None:
        for path in files:
            if path.name == filename:
                return path
        return None

    def open_background(self) -> None:
        path = filedialog.askopenfilename(
            title="打开背景图",
            initialdir=str(BACKGROUND_DIR),
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")],
        )
        if path:
            self.load_background(Path(path))
            self.log_action("打开背景图")

    def open_foreground(self) -> None:
        path = filedialog.askopenfilename(
            title="打开目标前景图",
            initialdir=str(FOREGROUND_DIR),
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")],
        )
        if path:
            self.load_foreground(Path(path))
            self.log_action("打开前景图")

    def load_background(self, path: Path) -> None:
        try:
            self.background_rgb = read_rgb(path)
        except OSError as exc:
            messagebox.showerror("读取失败", f"无法读取背景图:\n{path}\n\n{exc}")
            return
        self.background_path = path
        self.resize_canvas_for_image(self.background_rgb)
        self.result_rgb = self.background_rgb.copy()
        self.green_background_rgb = None
        self.current_mask = None
        self.mapped_rgb = None
        self.mapped_mask = None
        self.transformed_preview_rgb = None
        self.transformed_preview_mask = None
        self.load_time = time.perf_counter()
        self.status_var.set(f"已读取背景图: {path.name}")
        self.points.clear()
        self.brush_map = None
        self.brush_enabled = False
        self.refresh_brush_button()
        self.reset_cylinder_arc_controls()
        self.undo_stack.clear()
        self.interaction_count = 0
        self.count_var.set("交互次数 0")
        self.refresh_mapping_after_file_load()

    def load_foreground(self, path: Path) -> None:
        try:
            rgb = read_rgb(path)
        except OSError as exc:
            messagebox.showerror("读取失败", f"无法读取前景图:\n{path}\n\n{exc}")
            return
        self.foreground_path = path
        self.foreground_mask = make_foreground_mask(path, rgb)
        self.foreground_rgb = bleed_edge_colors(rgb, self.foreground_mask)
        self.mapped_rgb = None
        self.mapped_mask = None
        self.transformed_preview_rgb = None
        self.transformed_preview_mask = None
        self.brush_map = None
        self.status_var.set(f"已读取前景图: {path.name}")
        self.refresh_mapping_after_file_load()

    def refresh_mapping_after_file_load(self) -> None:
        self.redraw()

    def resize_canvas_for_image(self, image_rgb: np.ndarray) -> None:
        height, width = image_rgb.shape[:2]
        if width <= 0 or height <= 0:
            return
        display_width = max(CANVAS_MIN_WIDTH, CANVAS_TARGET_WIDTH)
        self.source_canvas.configure(width=display_width, height=CANVAS_TARGET_HEIGHT)
        self.root.geometry(f"{display_width + WINDOW_EXTRA_WIDTH}x{CANVAS_TARGET_HEIGHT + WINDOW_EXTRA_HEIGHT + 220}")
        self.root.update_idletasks()

    def set_mode(self, mode: str) -> None:
        if mode == self.mode_var.get():
            return
        self.mode_var.set(mode)
        self.change_mode()

    def change_mode(self) -> None:
        self.log_action(f"切换模式: {self.mode_var.get()}")
        self.refresh_mode_buttons()
        self.current_mask = None
        self.mapped_rgb = None
        self.mapped_mask = None
        self.transformed_preview_rgb = None
        self.transformed_preview_mask = None
        self.brush_map = None
        self.green_background_rgb = None
        self.brush_enabled = False
        self.refresh_brush_button()
        self.result_rgb = None if self.background_rgb is None else self.background_rgb.copy()
        self.points.clear()
        self.reset_cylinder_arc_controls()
        self.undo_stack.clear()
        self.status_var.set("请重新人工点击 4 个角点选择贴图位置")
        self.redraw()

    def refresh_mode_buttons(self) -> None:
        mode = self.mode_var.get()
        self.mode_text_var.set(f"当前模式：{'平面贴图' if mode == 'laptop' else '柱面贴图'}")
        for button, value in ((getattr(self, "laptop_button", None), "laptop"), (getattr(self, "cylinder_button", None), "cylinder")):
            if button is None:
                continue
            active = mode == value
            button.configure(
                bg=ACTIVE_BLUE if active else BUTTON_BG,
                fg="#ffffff" if active else "#29476b",
                activebackground="#1f6fd4" if active else "#eef5ff",
                activeforeground="#ffffff" if active else "#29476b",
            )

    def parameter_changed(self, action: str, preview_only: bool = False) -> None:
        if not preview_only:
            self.finish_parameter_edit(action)
            return
        self.apply_mapping(count_interaction=False)

    def finish_parameter_edit(self, action: str) -> None:
        if len(self.points) != 4:
            self.log_action(action)
            self.redraw()
            return
        self.log_action(action)
        self.apply_mapping(count_interaction=False)

    def refresh_brush_button(self) -> None:
        if not hasattr(self, "brush_button"):
            return
        if self.brush_enabled:
            self.brush_button.configure(text="关闭混合笔刷", bg=ACTIVE_BLUE, fg="#ffffff", activebackground="#1f6fd4")
        else:
            self.brush_button.configure(text="启用混合笔刷", bg=BUTTON_BG, fg="#29476b", activebackground="#eef5ff")

    def toggle_blend_brush(self) -> None:
        self.brush_enabled = not self.brush_enabled
        self.refresh_brush_button()
        self.source_canvas.configure(cursor="crosshair" if self.brush_enabled else "")
        self.status_var.set("混合笔刷已启用：在前景边缘拖动" if self.brush_enabled else "混合笔刷已关闭")

    def paint_blend_brush(self, point: tuple[float, float]) -> None:
        if self.background_rgb is None or self.current_mask is None:
            return
        height, width = self.current_mask.shape[:2]
        x = int(round(point[0]))
        y = int(round(point[1]))
        radius = max(2, int(self.brush_size_var.get() / 2))
        x1, x2 = max(0, x - radius), min(width, x + radius + 1)
        y1, y2 = max(0, y - radius), min(height, y + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return
        if self.brush_map is None or self.brush_map.shape != (height, width):
            self.brush_map = np.zeros((height, width), dtype=np.uint8)
        yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        stamp = np.clip(1.0 - distance / max(radius, 1), 0.0, 1.0)
        stamp = stamp * stamp * (3.0 - 2.0 * stamp)
        stamp *= self.brush_strength_var.get() / 100.0
        self.brush_map[y1:y2, x1:x2] = np.maximum(
            self.brush_map[y1:y2, x1:x2], np.round(stamp * 255.0).astype(np.uint8)
        )

    def apply_brush_result(self) -> None:
        if self.result_rgb is None or self.background_rgb is None or self.current_mask is None:
            return
        self.result_rgb = apply_edge_brush(self.result_rgb, self.background_rgb, self.current_mask, self.brush_map)

    def clear_points(self) -> None:
        self.push_undo()
        self.points.clear()
        self.reset_cylinder_arc_controls()
        self.current_mask = None
        self.mapped_rgb = None
        self.mapped_mask = None
        self.transformed_preview_rgb = None
        self.transformed_preview_mask = None
        self.brush_map = None
        self.green_background_rgb = None
        self.log_action("清空控制点")
        self.reset_result(count_interaction=False)
        self.redraw()

    def snapshot_state(self) -> UndoState:
        result_copy = None if self.result_rgb is None else self.result_rgb.copy()
        mask_copy = None if self.current_mask is None else self.current_mask.copy()
        return UndoState(
            result_copy,
            mask_copy,
            None if self.brush_map is None else self.brush_map.copy(),
            list(self.points),
            self.interaction_count,
            self.wrap_var.get(),
            self.cover_var.get(),
            self.light_var.get(),
            self.texture_var.get(),
            self.opacity_var.get(),
            self.feather_var.get(),
            self.brightness_adjust_var.get(),
            self.hue_shift_var.get(),
            self.temperature_var.get(),
            self.saturation_var.get(),
            self.contrast_var.get(),
            self.blend_var.get(),
            self.brightness_var.get(),
            self.cylinder_top_arc_factor,
            self.cylinder_bottom_arc_factor,
        )

    def push_undo(self, state: UndoState | None = None) -> None:
        self.undo_stack.append(state or self.snapshot_state())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def reset_cylinder_arc_controls(self) -> None:
        self.cylinder_top_arc_factor = 1.0
        self.cylinder_bottom_arc_factor = 1.68

    def restore_state(self, state: UndoState) -> None:
        self.result_rgb = None if state.result_rgb is None else state.result_rgb.copy()
        self.current_mask = None if state.current_mask is None else state.current_mask.copy()
        self.mapped_rgb = None
        self.mapped_mask = None
        self.transformed_preview_rgb = None
        self.transformed_preview_mask = None
        self.brush_map = None if state.brush_map is None else state.brush_map.copy()
        self.points = list(state.points)
        self.interaction_count = state.interaction_count
        self.wrap_var.set(state.wrap)
        self.cover_var.set(state.cover)
        self.light_var.set(state.light)
        self.texture_var.set(state.texture)
        self.opacity_var.set(state.opacity)
        self.feather_var.set(state.feather)
        self.brightness_adjust_var.set(state.brightness_adjust)
        self.hue_shift_var.set(state.hue_shift)
        self.temperature_var.set(state.temperature)
        self.saturation_var.set(state.saturation)
        self.contrast_var.set(state.contrast)
        self.blend_var.set(state.blend)
        self.brightness_var.set(state.brightness)
        self.cylinder_top_arc_factor = state.top_arc_factor
        self.cylinder_bottom_arc_factor = state.bottom_arc_factor
        self.count_var.set(f"交互次数 {self.interaction_count}")

    def reset_color_adjustments(self) -> None:
        self.hue_shift_var.set(0)
        self.temperature_var.set(0)
        self.saturation_var.set(100)
        self.contrast_var.set(100)
        self.parameter_changed("重置前景调色")

    def fuse_image(self) -> None:
        self.apply_mapping(count_interaction=True)

    def prepare_green_background(self) -> None:
        if self.background_rgb is None or self.mode_var.get() != "laptop" or len(self.points) != 4:
            self.green_background_rgb = None
            return
        detection_points = expand_quad(self.points, factor=1.65)
        self.green_background_rgb = replace_white_cat_with_green_screen(
            self.background_rgb.copy(), detection_points
        )

    def apply_mapping(self, count_interaction: bool = False, undo_state: UndoState | None = None) -> None:
        if self.background_rgb is None or self.foreground_rgb is None or self.foreground_mask is None:
            return
        if len(self.points) != 4:
            self.status_var.set("请在原图上依次点击 4 个控制点")
            return
        if count_interaction:
            self.push_undo(undo_state)

        background = self.background_rgb.copy()
        out_h, out_w = background.shape[:2]
        mode = self.mode_var.get()
        if mode == "laptop":
            if self.green_background_rgb is None:
                self.prepare_green_background()
            cleaned_background = self.green_background_rgb if self.green_background_rgb is not None else background
            mapped, mapped_mask = self.map_laptop(out_w, out_h)
            # 前景的亮度、边缘颜色和梯度融合始终参考原始背景，避免绿色替换层污染前景。
            result = self.compose(background, mapped, mapped_mask)
            if np.any(cleaned_background != background):
                protect = feather_mask(mapped_mask, max(2, self.feather_var.get())).astype(np.float32) / 255.0
                result = (
                    cleaned_background.astype(np.float32) * (1.0 - protect[..., None])
                    + result.astype(np.float32) * protect[..., None]
                )
                result = np.clip(result, 0, 255).astype(np.uint8)
            self.result_rgb = result
            preview_rgb, preview_mask = mapped, mapped_mask
        else:
            mapped, mapped_mask, preview_rgb, preview_mask = self.map_cylinder(out_w, out_h)
            self.result_rgb = self.compose(background, mapped, mapped_mask)

        self.current_mask = mapped_mask.copy()
        self.mapped_rgb = mapped.copy()
        self.mapped_mask = mapped_mask.copy()
        self.transformed_preview_rgb = preview_rgb.copy()
        self.transformed_preview_mask = preview_mask.copy()
        self.apply_brush_result()
        if count_interaction:
            self.record_interaction("完成一次贴图融合")
        self.status_var.set(f"已生成 {mode} 贴图结果")
        self.redraw_result()

    def map_laptop(self, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.foreground_rgb is not None
        assert self.foreground_mask is not None
        foreground, mask = crop_to_mask(self.foreground_rgb, self.foreground_mask)
        foreground = bleed_edge_colors(foreground, mask, iterations=24)
        fg_h, fg_w = foreground.shape[:2]
        src = np.array([[0, 0], [fg_w - 1, 0], [fg_w - 1, fg_h - 1], [0, fg_h - 1]], dtype=np.float32)
        dst = np.array(order_points(self.effective_points()), dtype=np.float32)
        warped_rgb = warp_perspective(foreground, src, dst, (out_w, out_h), is_mask=False)
        warped_mask = warp_perspective(mask, src, dst, (out_w, out_h), is_mask=True)
        return warped_rgb.astype(np.uint8), clean_mask(warped_mask.astype(np.uint8), kernel_size=3)

    def map_cylinder(
        self,
        out_w: int,
        out_h: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.foreground_rgb is not None
        assert self.foreground_mask is not None
        assert self.background_rgb is not None
        instance_quads = cylinder_instance_quads(self.effective_points(), (out_w, out_h))
        layers: list[tuple[np.ndarray, np.ndarray]] = []
        for quad in instance_quads:
            surface = cylinder_surface_from_points(
                quad,
                (out_w, out_h),
                self.cover_var.get() / 100.0,
                self.wrap_var.get() / 100.0,
                self.cylinder_top_arc_factor,
                self.cylinder_bottom_arc_factor,
            )
            box = surface.box
            box_width = max(2, box[2] - box[0] + 1)
            box_height = max(2, box[3] - box[1] + 1)
            foreground, mask = fit_foreground_stretch_canvas(
                self.foreground_rgb,
                self.foreground_mask,
                box_width,
                box_height,
            )
            layers.append(
                cylinder_warp(
                    foreground,
                    mask,
                    surface,
                    (out_w, out_h),
                    self.wrap_var.get() / 100.0,
                    self.light_var.get() / 100.0,
                    self.texture_var.get() / 100.0,
                    self.background_rgb,
                )
            )

        combined_rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
        combined_mask = np.zeros((out_h, out_w), dtype=np.float32)
        for layer_rgb, layer_mask in layers:
            new_alpha = np.clip(layer_mask.astype(np.float32) / 255.0, 0.0, 1.0)
            old_alpha = np.clip(combined_mask / 255.0, 0.0, 1.0)
            output_alpha = new_alpha + old_alpha * (1.0 - new_alpha)
            numerator = combined_rgb * old_alpha[..., None] * (1.0 - new_alpha[..., None])
            numerator += layer_rgb.astype(np.float32) * new_alpha[..., None]
            valid = output_alpha > 1e-5
            combined_rgb[valid] = numerator[valid] / output_alpha[valid, None]
            combined_rgb[~valid] = 0.0
            combined_mask = output_alpha * 255.0

        mapped_rgb = np.clip(combined_rgb, 0, 255).astype(np.uint8)
        mapped_mask = np.clip(combined_mask, 0, 255).astype(np.uint8)
        bottom_rgb, bottom_mask = layers[-1]
        return mapped_rgb, mapped_mask, bottom_rgb.astype(np.uint8), bottom_mask.astype(np.uint8)

    def compose(self, background: np.ndarray, foreground: np.ndarray, mask: np.ndarray) -> np.ndarray:
        blend = self.blend_var.get()
        opacity = self.opacity_var.get() / 100.0
        feather = self.feather_var.get() if blend != "普通融合" else 0
        if self.mode_var.get() == "cylinder":
            opacity = min(opacity, 0.97)
            feather = max(feather, 2)
        alpha = feather_mask(mask, feather).astype(np.float32) / 255.0
        alpha *= float(np.clip(opacity, 0.0, 1.0))
        prepared = foreground.astype(np.uint8)
        if self.brightness_var.get() and self.mode_var.get() != "cylinder":
            prepared = match_brightness(prepared, background, (alpha * 255).astype(np.uint8))
        prepared = adjust_brightness(prepared, self.brightness_adjust_var.get())
        prepared = apply_foreground_color_adjustments(
            prepared,
            hue_shift=self.hue_shift_var.get(),
            temperature=self.temperature_var.get(),
            saturation=self.saturation_var.get(),
            contrast=self.contrast_var.get(),
        )
        prepared = adapt_edge_colors_to_background(prepared, background, alpha)
        result = background.astype(np.float32) * (1.0 - alpha[..., None]) + prepared.astype(np.float32) * alpha[..., None]
        result = np.clip(result, 0, 255).astype(np.uint8)
        if blend == "梯度融合" and self.mode_var.get() != "cylinder":
            gradient = try_gradient_edge_blend(background, prepared, result, mask, feather)
            if gradient is not None:
                return gradient
            self.status_var.set("边缘梯度融合失败，已自动使用羽化融合")
        elif blend == "Poisson融合":
            poisson = try_poisson_blend(background, prepared, np.where(mask > 10, 255, 0).astype(np.uint8))
            if poisson is not None:
                return poisson
            self.status_var.set("Poisson融合失败，已自动使用羽化融合")
        return result

    def reset_result(self, count_interaction: bool = True) -> None:
        if self.background_rgb is not None:
            self.result_rgb = self.background_rgb.copy()
        self.current_mask = None
        self.mapped_rgb = None
        self.mapped_mask = None
        self.transformed_preview_rgb = None
        self.transformed_preview_mask = None
        self.brush_map = None
        self.load_time = time.perf_counter()
        self.interaction_count = 0
        self.count_var.set("交互次数 0")
        if count_interaction:
            self.log_action("重置结果")
        self.redraw_result()

    def undo(self) -> None:
        if not self.undo_stack:
            self.status_var.set("没有可撤销的步骤")
            return
        state = self.undo_stack.pop()
        self.restore_state(state)
        self.log_action("撤销")
        self.redraw()

    def save_result(self) -> None:
        if self.result_rgb is None:
            messagebox.showwarning("没有结果", "请先点击融合图像。")
            return
        RESULT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = RESULT_DIR / f"result_{self.mode_var.get()}_{timestamp}.png"
        gui_path = RESULT_DIR / f"workbench_{self.mode_var.get()}_{timestamp}.png"
        save_rgb(result_path, self.result_rgb)
        if not self.capture_gui_snapshot(gui_path):
            self.status_var.set(f"结果图已保存，但工作台截图失败: {result_path.name}")
            self.log_action("保存结果图（工作台截图失败）")
            return
        self.log_action("保存结果图和工作台截图")
        self.status_var.set(f"已保存结果图和工作台截图: {result_path.name} / {gui_path.name}")

    def capture_gui_snapshot(self, path: Path) -> bool:
        self.root.update_idletasks()
        try:
            x = self.root.winfo_rootx()
            y = self.root.winfo_rooty()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            snapshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            snapshot.save(path)
        except Exception as exc:
            messagebox.showerror("截图失败", f"无法保存 GUI 截图:\n{exc}")
            return False
        return True

    def save_gui_snapshot(self) -> None:
        RESULT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULT_DIR / f"gui_{self.mode_var.get()}_{timestamp}.png"
        if not self.capture_gui_snapshot(path):
            return
        self.log_action("保存GUI截图")
        self.status_var.set(f"GUI 截图已保存: {path.name}")

    def record_interaction(self, action: str) -> None:
        self.interaction_count += 1
        self.count_var.set(f"交互次数 {self.interaction_count}")
        self.log_action(action)

    def log_action(self, action: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        line = f"[{now}] {action}\n"
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, line)
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def cylinder_arc_base_metrics(self) -> tuple[float, float, float, float] | None:
        if self.background_rgb is None or len(self.points) != 4 or self.mode_var.get() != "cylinder":
            return None
        tl, tr, br, bl = order_points(self.effective_points())
        left_x = (tl[0] + bl[0]) / 2.0
        right_x = (tr[0] + br[0]) / 2.0
        selected_width = max(2.0, abs(right_x - left_x))
        center_x = (left_x + right_x) / 2.0
        top_mid = (tl[1] + tr[1]) / 2.0
        bottom_mid = (bl[1] + br[1]) / 2.0
        avg_height = max(8.0, bottom_mid - top_mid)
        bend = float(np.clip(self.wrap_var.get() / 100.0, 0.0, 1.0))
        arc_depth = min(avg_height * 0.16, selected_width * 0.13) * (0.55 + 0.65 * bend)
        return center_x, top_mid, bottom_mid, max(1.0, arc_depth)

    def cylinder_arc_handle_points(self) -> dict[str, tuple[float, float]]:
        metrics = self.cylinder_arc_base_metrics()
        if metrics is None:
            return {}
        center_x, top_mid, bottom_mid, arc_depth = metrics
        return {
            "arc_top": (center_x, top_mid - arc_depth * 0.72 * self.cylinder_top_arc_factor),
            "arc_bottom": (center_x, bottom_mid + arc_depth * 1.05 * self.cylinder_bottom_arc_factor),
        }

    def find_nearest_arc_handle(self, point: tuple[float, float]) -> str | None:
        handles = self.cylinder_arc_handle_points()
        if not handles:
            return None
        threshold = 16.0 / max(self.display_geometry.scale, 0.01)
        nearest_name = min(handles, key=lambda name: math.dist(point, handles[name]))
        if math.dist(point, handles[nearest_name]) <= threshold:
            return nearest_name
        return None

    def update_cylinder_arc_from_point(self, handle_name: str, point: tuple[float, float]) -> None:
        metrics = self.cylinder_arc_base_metrics()
        if metrics is None:
            return
        _center_x, top_mid, bottom_mid, arc_depth = metrics
        if handle_name == "arc_top":
            self.cylinder_top_arc_factor = float(np.clip((top_mid - point[1]) / max(1.0, arc_depth * 0.72), 0.0, 2.2))
        elif handle_name == "arc_bottom":
            self.cylinder_bottom_arc_factor = float(
                np.clip((point[1] - bottom_mid) / max(1.0, arc_depth * 1.05), 0.0, 2.5)
            )

    def _on_canvas_press(self, event: tk.Event) -> None:
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        if self.brush_enabled:
            if self.current_mask is None:
                self.status_var.set("请先完成贴图，再使用混合笔刷")
                return
            self.push_undo()
            self.brush_stroking = True
            self.paint_blend_brush(point)
            self.apply_mapping(count_interaction=False)
            return
        nearest = self.find_nearest_point(point)
        if nearest is not None:
            self.drag_index = nearest
            self.drag_mode = "point"
            self.drag_anchor = point
            self.drag_original_points = list(self.points)
            self.drag_started = False
            return
        arc_handle = self.find_nearest_arc_handle(point)
        if arc_handle is not None:
            self.drag_index = None
            self.drag_mode = arc_handle
            self.drag_anchor = point
            self.drag_original_points = list(self.points)
            self.drag_started = False
            return
        if len(self.points) == 4 and self.is_point_in_current_mask(point):
            self.drag_index = None
            self.drag_mode = "mask"
            self.drag_anchor = point
            self.drag_original_points = list(self.points)
            self.drag_started = False
            return
        if len(self.points) < 4:
            self.points.append(point)
            self.log_action(f"添加控制点 {len(self.points)}")
            if len(self.points) == 4:
                self.points = order_points(self.points)
                self.prepare_green_background()
                self.apply_mapping(count_interaction=False)
            self.redraw()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self.brush_stroking:
            point = self.canvas_to_image_clamped(event.x, event.y)
            if point is not None:
                self.paint_blend_brush(point)
                self.apply_mapping(count_interaction=False)
            return
        if self.drag_mode is None:
            return
        point = self.canvas_to_image_clamped(event.x, event.y)
        if point is None:
            return
        if self.drag_mode == "point" and self.drag_index is not None:
            self.points[self.drag_index] = self.inverse_effective_point(point)
        elif self.drag_mode in ("arc_top", "arc_bottom"):
            self.update_cylinder_arc_from_point(self.drag_mode, point)
        elif self.drag_mode == "mask" and self.drag_anchor is not None:
            dx = point[0] - self.drag_anchor[0]
            dy = point[1] - self.drag_anchor[1]
            self.points = self.translated_points(dx, dy)
        self.drag_started = True
        if len(self.points) == 4:
            self.apply_mapping(count_interaction=False)
        else:
            self.redraw()

    def _on_canvas_release(self, _event: tk.Event) -> None:
        if self.brush_stroking:
            self.brush_stroking = False
            self.log_action("使用混合笔刷")
            self.status_var.set("混合笔刷已应用，可继续拖动或点击撤销")
            return
        if self.drag_mode is not None and self.drag_started:
            self.points = order_points(self.points)
            action = "调整弧形边" if self.drag_mode in ("arc_top", "arc_bottom") else "调整贴图位置"
            self.log_action(action)
            self.apply_mapping(count_interaction=False)
        self.clear_drag_state()

    def _on_canvas_double_click(self, event: tk.Event) -> str | None:
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return None
        nearest = self.find_nearest_point(point)
        if nearest is None:
            return None

        self.push_undo()
        self.points.clear()
        self.current_mask = None
        self.mapped_rgb = None
        self.mapped_mask = None
        self.transformed_preview_rgb = None
        self.transformed_preview_mask = None
        if self.background_rgb is not None:
            self.result_rgb = self.background_rgb.copy()
        self.status_var.set("已清空控制点，请重新人工选择贴图位置")
        self.log_action("双击清空控制点")
        self.clear_drag_state()
        self.redraw()
        return "break"

    def clear_drag_state(self) -> None:
        self.drag_index = None
        self.drag_mode = None
        self.drag_anchor = None
        self.drag_original_points = []
        self.drag_started = False

    def find_nearest_point(self, point: tuple[float, float]) -> int | None:
        if not self.points:
            return None
        threshold = 14.0 / max(self.display_geometry.scale, 0.01)
        distances = [math.dist(point, current) for current in self.effective_points()]
        index = int(np.argmin(distances))
        if distances[index] <= threshold:
            return index
        return None

    def effective_points(self) -> list[tuple[float, float]]:
        return list(self.points)

    def inverse_effective_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return point

    def is_point_in_current_mask(self, point: tuple[float, float]) -> bool:
        if self.current_mask is None:
            return False
        x = int(round(point[0]))
        y = int(round(point[1]))
        height, width = self.current_mask.shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        radius = max(3, int(8.0 / max(self.display_geometry.scale, 0.01)))
        x1 = max(0, x - radius)
        x2 = min(width, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(height, y + radius + 1)
        return bool(np.any(self.current_mask[y1:y2, x1:x2] > 20))

    def translated_points(self, dx: float, dy: float) -> list[tuple[float, float]]:
        if self.background_rgb is None:
            return list(self.drag_original_points)
        height, width = self.background_rgb.shape[:2]
        moved = [(x + dx, y + dy) for x, y in self.drag_original_points]
        min_x = min(x for x, _y in moved)
        max_x = max(x for x, _y in moved)
        min_y = min(y for _x, y in moved)
        max_y = max(y for _x, y in moved)
        adjust_x = 0.0
        adjust_y = 0.0
        if min_x < 0:
            adjust_x = -min_x
        elif max_x > width - 1:
            adjust_x = width - 1 - max_x
        if min_y < 0:
            adjust_y = -min_y
        elif max_y > height - 1:
            adjust_y = height - 1 - max_y
        return [(float(x + adjust_x), float(y + adjust_y)) for x, y in moved]

    def canvas_to_image(self, x: int, y: int) -> tuple[float, float] | None:
        if self.background_rgb is None:
            return None
        geom = self.display_geometry
        ix = (x - geom.offset_x) / max(geom.scale, 1e-6)
        iy = (y - geom.offset_y) / max(geom.scale, 1e-6)
        height, width = self.background_rgb.shape[:2]
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            return None
        return float(ix), float(iy)

    def canvas_to_image_clamped(self, x: int, y: int) -> tuple[float, float] | None:
        if self.background_rgb is None:
            return None
        geom = self.display_geometry
        ix = (x - geom.offset_x) / max(geom.scale, 1e-6)
        iy = (y - geom.offset_y) / max(geom.scale, 1e-6)
        height, width = self.background_rgb.shape[:2]
        return float(np.clip(ix, 0, width - 1)), float(np.clip(iy, 0, height - 1))

    def image_to_canvas(self, point: tuple[float, float]) -> tuple[int, int]:
        geom = self.display_geometry
        return int(point[0] * geom.scale + geom.offset_x), int(point[1] * geom.scale + geom.offset_y)

    def redraw(self) -> None:
        self.redraw_source()

    def redraw_source(self) -> None:
        self.source_canvas.delete("all")
        if self.background_rgb is None:
            self._draw_empty(self.source_canvas, "请打开背景图")
            self.redraw_previews()
            return
        base = self.result_rgb if self.result_rgb is not None else self.background_rgb
        image = Image.fromarray(self.annotate_image(base), mode="RGB").convert("RGBA")
        self._draw_points_on_image(image)
        self.display_geometry, self.canvas_photo = self._paint_image(self.source_canvas, image.convert("RGB"))
        self.redraw_previews()

    def preview_image(self, rgb: np.ndarray | None, mask: np.ndarray | None = None) -> np.ndarray | None:
        if rgb is None:
            return None
        if mask is not None and np.any(mask > 0):
            rgb, mask = crop_to_mask(rgb, mask, padding=2)
        if mask is None:
            return rgb.astype(np.uint8)

        height, width = mask.shape
        yy, xx = np.mgrid[0:height, 0:width]
        checker = np.where(((xx // 16) + (yy // 16)) % 2 == 0, 238, 216).astype(np.uint8)
        checker_rgb = np.repeat(checker[:, :, None], 3, axis=2).astype(np.float32)
        alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
        composed = checker_rgb * (1.0 - alpha[..., None]) + rgb.astype(np.float32) * alpha[..., None]
        return np.clip(composed, 0, 255).astype(np.uint8)

    def redraw_previews(self) -> None:
        if not hasattr(self, "foreground_preview_canvas"):
            return
        for canvas in (self.foreground_preview_canvas, self.mapped_preview_canvas):
            canvas.delete("all")

        foreground_preview = self.preview_image(self.foreground_rgb, self.foreground_mask)
        mapped_preview = self.preview_image(self.transformed_preview_rgb, self.transformed_preview_mask)
        if foreground_preview is None:
            self._draw_empty(self.foreground_preview_canvas, "未加载目标图像")
        else:
            _, self.foreground_preview_photo = self._paint_image(
                self.foreground_preview_canvas,
                Image.fromarray(foreground_preview, mode="RGB"),
            )
        if mapped_preview is None:
            self._draw_empty(self.mapped_preview_canvas, "完成四点选择后显示")
        else:
            _, self.mapped_preview_photo = self._paint_image(
                self.mapped_preview_canvas,
                Image.fromarray(mapped_preview, mode="RGB"),
            )

    def redraw_result(self) -> None:
        self.redraw_source()

    def annotated_result(self) -> np.ndarray:
        image = self.result_rgb if self.result_rgb is not None else self.background_rgb
        if image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return self.annotate_image(image)

    def annotate_image(self, image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image, mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(pil, mode="RGBA")
        elapsed = 0.0 if self.load_time is None else time.perf_counter() - self.load_time
        text = f"Time: {elapsed:.1f}s   Interactions: {self.interaction_count}   Mode: {self.mode_var.get()}"
        bbox = draw.textbbox((0, 0), text)
        draw.rectangle((12, 12, 24 + bbox[2], 36 + bbox[3]), fill=(0, 0, 0, 165))
        draw.text((18, 18), text, fill=(255, 255, 255, 255))
        return np.array(pil.convert("RGB"), dtype=np.uint8)

    def _dashed_line(
        self,
        draw: ImageDraw.ImageDraw,
        start: tuple[int, int],
        end: tuple[int, int],
        fill: tuple[int, int, int, int],
        dash: int = 8,
        gap: int = 7,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 0:
            return
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        position = 0.0
        while position < length:
            segment_end = min(position + dash, length)
            draw.line(
                (
                    x1 + dx * position,
                    y1 + dy * position,
                    x1 + dx * segment_end,
                    y1 + dy * segment_end,
                ),
                fill=fill,
                width=2,
            )
            position += dash + gap

    def _draw_points_on_image(self, image: Image.Image) -> None:
        draw = ImageDraw.Draw(image, mode="RGBA")
        if len(self.points) == 4:
            if self.mode_var.get() == "cylinder":
                surface = cylinder_surface_from_points(
                    self.effective_points(),
                    image.size,
                    self.cover_var.get() / 100.0,
                    self.wrap_var.get() / 100.0,
                    self.cylinder_top_arc_factor,
                    self.cylinder_bottom_arc_factor,
                )
                boundary = cylinder_surface_polygon(surface)
                if len(boundary) > 2:
                    draw.line(boundary + [boundary[0]], fill=(47, 128, 237, 230), width=3)
            else:
                ordered = order_points(self.effective_points())
                flat = [(float(x), float(y)) for x, y in ordered]
                for start, end in zip(flat, flat[1:] + [flat[0]]):
                    self._dashed_line(draw, start, end, (47, 128, 237, 230), dash=9, gap=6)

        labels = ["1", "2", "3", "4"]
        for index, (x, y) in enumerate(self.effective_points()):
            r = 10
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(47, 128, 237, 255), outline=(255, 255, 255, 255), width=3)
            draw.text((x - 4, y - 7), labels[index], fill=(255, 255, 255, 255))

        if len(self.points) == 4 and self.mode_var.get() == "cylinder":
            for label, (x, y) in zip(("5", "6"), self.cylinder_arc_handle_points().values()):
                r = 9
                draw.ellipse(
                    (x - r, y - r, x + r, y + r),
                    fill=(220, 104, 3, 255),
                    outline=(255, 255, 255, 255),
                    width=3,
                )
                draw.text((x - 4, y - 7), label, fill=(255, 255, 255, 255))

    def _paint_image(self, canvas: tk.Canvas, image: Image.Image) -> tuple[DisplayGeometry, ImageTk.PhotoImage]:
        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        scale = min(canvas_width / image.width, canvas_height / image.height)
        display_width = max(1, int(image.width * scale))
        display_height = max(1, int(image.height * scale))
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        resized = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#eef2f6", outline="")
        canvas.create_image(offset_x, offset_y, image=photo, anchor=tk.NW)
        return DisplayGeometry(scale, offset_x, offset_y, display_width, display_height), photo

    def _draw_empty(self, canvas: tk.Canvas, text: str) -> None:
        canvas.create_rectangle(0, 0, canvas.winfo_width(), canvas.winfo_height(), fill="#eef2f6", outline="")
        canvas.create_text(
            max(1, canvas.winfo_width() // 2),
            max(1, canvas.winfo_height() // 2),
            text=text,
            fill=MUTED,
            font=(UI_FONT, 16),
        )

    def _tick(self) -> None:
        if self.load_time is None:
            self.time_var.set("操作时间 0.0 s")
        else:
            elapsed = time.perf_counter() - self.load_time
            self.time_var.set(f"操作时间 {elapsed:.1f} s")
        self.root.after(200, self._tick)


def main() -> int:
    root = tk.Tk()
    PasteMappingApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
