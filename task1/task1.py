from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc: 
    raise SystemExit("Tkinter is required to run this desktop application.") from exc


GC_BGD = 0
GC_FGD = 1
GC_PR_BGD = 2
GC_PR_FGD = 3
MANUAL_NONE = 255

MARK_MODES = {
    "前景": GC_FGD,
    "背景": GC_BGD,
    "可能前景": GC_PR_FGD,
    "可能背景": GC_PR_BGD,
    "初始区域": GC_PR_FGD,
}

TOOLS = (
    "brush",
    "eraser",
    "line",
    "rectangle",
    "square",
    "ellipse",
    "circle",
    "pentagon",
    "hexagon",
    "polygon",
)

TOOL_LABELS = {
    "eraser": "橡皮擦",
    "brush": "涂抹",
    "line": "直线",
    "rectangle": "长方形",
    "square": "正方形",
    "ellipse": "椭圆",
    "circle": "圆",
    "pentagon": "五边形",
    "hexagon": "六边形",
    "polygon": "任意多边形",
}

UI_FONT_FAMILY = "Microsoft YaHei"
TASK1_DIR = Path(__file__).resolve().parent
ANNOTATION_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyh.ttf"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
)


def load_annotation_font(size: int = 16) -> ImageFont.ImageFont:
    # 图片上的时间、交互次数标注使用微软雅黑，避免 PIL 默认位图字体放大后发虚。
    for font_path in ANNOTATION_FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    try:
        return ImageFont.truetype("msyh.ttc", size)
    except OSError:
        return ImageFont.load_default()


def require_cv2():
    try:
        import cv2  # 类型检查忽略
    except ImportError as exc:
        raise RuntimeError(
            "缺少 OpenCV"
        ) from exc
    return cv2


def bleed_edge_colors(rgb: np.ndarray, alpha: np.ndarray, iterations: int = 16) -> np.ndarray:
    cv2 = require_cv2()
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


def read_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def write_rgb_image(path: str | Path, image_rgb: np.ndarray) -> None:
    Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").save(path)

#没有前景标记时的默认前景区域
def make_default_mask(width: int, height: int) -> np.ndarray:
    mask = np.full((height, width), GC_BGD, dtype=np.uint8)
    margin_x = max(1, int(width * 0.08))
    margin_y = max(1, int(height * 0.08))
    mask[margin_y : height - margin_y, margin_x : width - margin_x] = GC_PR_FGD
    return mask


def binary_from_grabcut_mask(mask: np.ndarray) -> np.ndarray:
    return np.where((mask == GC_FGD) | (mask == GC_PR_FGD), 255, 0).astype(np.uint8)


def expand_initial_foreground_from_marks(
    mask: np.ndarray,
    visual_marks: np.ndarray | None = None,
    padding_ratio: float = 0.35,
    min_padding: int = 24,
    max_seed_area_ratio: float = 0.20,
) -> tuple[np.ndarray, bool]:
    """第一次分割前只有少量前景笔画时，自动扩展出可能前景候选区域。"""
    output = mask.astype(np.uint8, copy=True)
    foreground = (output == GC_FGD) | (output == GC_PR_FGD)
    if not np.any(foreground):
        return output, False

    height, width = output.shape
    seed_area_ratio = float(np.count_nonzero(foreground)) / float(height * width)
    if seed_area_ratio > max_seed_area_ratio:
        return output, False

    ys, xs = np.where(foreground)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    box_width = max(1, x2 - x1 + 1)
    box_height = max(1, y2 - y1 + 1)
    pad_x = max(int(min_padding), int(box_width * padding_ratio), int(width * 0.04))
    pad_y = max(int(min_padding), int(box_height * padding_ratio), int(height * 0.04))

    x1 = max(0, x1 - pad_x)
    x2 = min(width - 1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(height - 1, y2 + pad_y)

    candidate = np.zeros_like(output, dtype=bool)
    candidate[y1 : y2 + 1, x1 : x2 + 1] = True
    candidate &= ~foreground
    candidate &= output != GC_PR_BGD

    if visual_marks is not None:
        explicit_background = (visual_marks == GC_BGD) | (visual_marks == GC_PR_BGD)
        candidate &= ~explicit_background

    before = output.copy()
    output[candidate] = GC_PR_FGD
    return output, not np.array_equal(before, output)


def enhance_bright_line_details(
    image_rgb: np.ndarray,
    result_mask: np.ndarray,
    seed_mask: np.ndarray,
    alpha_mask: np.ndarray | None = None,
    blocked_mask: np.ndarray | None = None,
    search_kernel_size: int = 41,
    detail_kernel_size: int = 9,
    bright_threshold: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """在人工前景笔画附近增强胡须、发丝一类明亮细线。"""
    cv2 = require_cv2()
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an HxWx3 RGB image")
    if result_mask.shape != image_rgb.shape[:2] or seed_mask.shape != result_mask.shape:
        raise ValueError("mask shapes must match the image")

    seed = seed_mask.astype(bool)
    if not np.any(seed):
        base_alpha = result_mask if alpha_mask is None else alpha_mask
        empty = np.zeros_like(result_mask, dtype=np.uint8)
        return result_mask.copy(), base_alpha.copy(), empty, 0

    search_kernel_size = max(5, int(search_kernel_size))
    if search_kernel_size % 2 == 0:
        search_kernel_size += 1
    detail_kernel_size = max(3, int(detail_kernel_size))
    if detail_kernel_size % 2 == 0:
        detail_kernel_size += 1

    seed_u8 = (seed.astype(np.uint8) * 255)
    search_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (search_kernel_size, search_kernel_size))
    search_area = cv2.dilate(seed_u8, search_kernel, iterations=1) > 0
    if blocked_mask is not None:
        search_area &= ~blocked_mask.astype(bool)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    contrast = cv2.subtract(gray, blurred)
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (detail_kernel_size, detail_kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, line_kernel)
    edges = cv2.Canny(gray, 40, 120)

    _, bright_lines = cv2.threshold(tophat, int(bright_threshold), 255, cv2.THRESH_BINARY)
    contrast_lines = (contrast > max(8, bright_threshold // 2)) & (gray > 80)
    candidate = (
        (bright_lines > 0)
        | ((edges > 0) & (tophat > max(8, bright_threshold // 2)))
        | contrast_lines
    ) & search_area

    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    candidate_u8 = cv2.morphologyEx(candidate.astype(np.uint8) * 255, cv2.MORPH_CLOSE, connect_kernel)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
    near_seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    near_seed = cv2.dilate(seed_u8, near_seed_kernel, iterations=1) > 0

    filtered = np.zeros_like(candidate_u8)
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 2 or area > 8000:
            continue
        component = labels == label
        aspect = max(width, height) / max(1, min(width, height))
        fill_ratio = area / max(1, width * height)
        thin_component = aspect >= 2.0 or fill_ratio <= 0.45
        touches_seed = bool(np.any(component & near_seed))
        bright_fraction = float(np.count_nonzero((tophat > max(8, bright_threshold // 2)) & component)) / float(area)
        mean_contrast = float(np.mean(contrast[component]))
        dense_detail = touches_seed and (bright_fraction >= 0.08 or mean_contrast >= 6.0)
        if thin_component or dense_detail:
            filtered[component] = 255

    added = filtered > 0
    output_mask = result_mask.copy()
    output_mask[added] = 255

    output_alpha = result_mask.copy() if alpha_mask is None else alpha_mask.copy()
    line_alpha = np.clip(tophat[added].astype(np.uint16) * 4, 210, 255).astype(np.uint8)
    output_alpha[added] = np.maximum(output_alpha[added], line_alpha)
    output_alpha[output_mask == 0] = 0
    return output_mask.astype(np.uint8), output_alpha.astype(np.uint8), filtered.astype(np.uint8), int(np.count_nonzero(added))


def edge_watershed_segment(
    image_rgb: np.ndarray,
    grabcut_mask: np.ndarray,
    current_mask: np.ndarray | None = None,
    manual_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """基于 Canny 边缘和分水岭的 CPU 分割，用于和 GrabCut / 模型结果做消融对比。"""
    cv2 = require_cv2()
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an HxWx3 RGB image")
    if grabcut_mask.shape != image_rgb.shape[:2]:
        raise ValueError("grabcut_mask shape must match the image")

    height, width = image_rgb.shape[:2]
    image_area = max(1, height * width)

    if current_mask is not None and current_mask.shape == (height, width) and np.any(current_mask > 0):
        candidate = current_mask > 0
    else:
        candidate = (grabcut_mask == GC_FGD) | (grabcut_mask == GC_PR_FGD)
    if not np.any(candidate):
        candidate = make_default_mask(width, height) == GC_PR_FGD

    manual_fg = np.zeros((height, width), dtype=bool)
    manual_bg = np.zeros((height, width), dtype=bool)
    if manual_mask is not None and manual_mask.shape == (height, width):
        manual_fg = (manual_mask == GC_FGD) | (manual_mask == GC_PR_FGD)
        manual_bg = (manual_mask == GC_BGD) | (manual_mask == GC_PR_BGD)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    low = int(max(10, 0.66 * median))
    high = int(min(255, max(low + 20, 1.33 * median)))
    edges = cv2.Canny(gray, low, high)

    # 从当前候选前景内部提取较可靠的前景种子，避免矩形框整体直接变成前景。
    sure_fg = manual_fg.copy()
    candidate_u8 = candidate.astype(np.uint8) * 255
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
    for label in range(1, labels_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(8, image_area // 20000):
            continue
        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        max_distance = float(distance.max())
        if max_distance > 1.0:
            sure_fg |= distance >= max(1.0, max_distance * 0.35)
        else:
            sure_fg |= component

    if not np.any(sure_fg):
        sure_fg = candidate.copy()

    candidate_kernel_size = max(9, int(min(height, width) * 0.035))
    if candidate_kernel_size % 2 == 0:
        candidate_kernel_size += 1
    candidate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (candidate_kernel_size, candidate_kernel_size)
    )
    allowed_area = cv2.dilate(candidate_u8, candidate_kernel, iterations=1) > 0

    sure_bg = (~allowed_area) | manual_bg
    sure_bg[:2, :] = True
    sure_bg[-2:, :] = True
    sure_bg[:, :2] = True
    sure_bg[:, -2:] = True

    sure_fg &= ~sure_bg
    unknown = ~(sure_fg | sure_bg)

    fg_labels_count, markers = cv2.connectedComponents(sure_fg.astype(np.uint8), connectivity=8)
    markers = markers.astype(np.int32) + 1
    markers[sure_bg] = 1
    markers[unknown] = 0

    # 分水岭在梯度图上更容易沿真实边缘收敛。
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gradient[edges > 0] = 255
    watershed_source = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    cv2.watershed(watershed_source, markers)
    result = (markers > 1) & allowed_area
    result[manual_fg] = True
    result[manual_bg] = False

    if not np.any(result):
        result = candidate & ~manual_bg

    result_mask = (result.astype(np.uint8) * 255)
    return result_mask.astype(np.uint8), edges.astype(np.uint8), max(0, fg_labels_count - 1)


def postprocess_binary_mask(
    mask: np.ndarray,
    use_opening: bool = True,
    use_closing: bool = True,
    keep_largest: bool = False,
    smooth_edges: bool = True,
    kernel_size: int = 5,
) -> np.ndarray:
    cv2 = require_cv2()
    output = np.where(mask > 0, 255, 0).astype(np.uint8)
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if use_opening:
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    if use_closing:
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    if keep_largest:
        output = keep_largest_component(output)
    if smooth_edges:
        output = cv2.GaussianBlur(output, (kernel_size, kernel_size), 0)
        _, output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)
    return output.astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if labels_count <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def run_grabcut(image_rgb: np.ndarray, mask: np.ndarray, iterations: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """在 CPU 上运行一次 GrabCut，并返回更新后的 GrabCut 掩码和二值前景掩码。"""
    cv2 = require_cv2()
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an HxWx3 RGB image")

    working_mask = mask.astype(np.uint8, copy=True)
    if not np.any((working_mask == GC_FGD) | (working_mask == GC_PR_FGD)):
        height, width = working_mask.shape
        working_mask = make_default_mask(width, height)
        #没有背景标记时的边界背景保底
    if not np.any((working_mask == GC_BGD) | (working_mask == GC_PR_BGD)):
        working_mask[0, :] = GC_BGD
        working_mask[-1, :] = GC_BGD
        working_mask[:, 0] = GC_BGD
        working_mask[:, -1] = GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, working_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    return working_mask, binary_from_grabcut_mask(working_mask)


@dataclass
class DisplayGeometry:
    scale: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    display_width: int = 1
    display_height: int = 1


@dataclass
class MarkUndoState:
    grabcut_mask: np.ndarray | None
    result_mask: np.ndarray | None
    alpha_mask: np.ndarray | None
    manual_mask: np.ndarray | None
    visual_marks: np.ndarray | None
    pending_visual_marks: np.ndarray | None
    pending_edits: bool
    segmentation_count: int
    mark_count: int
    operation_seconds: float


class InteractiveSegmentationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("交互式图像分割")
        self.root.geometry("1320x820")
        self.root.minsize(1180, 720)

        self.image_path: Path | None = None
        self.image_rgb: np.ndarray | None = None
        self.grabcut_mask: np.ndarray | None = None
        self.result_mask: np.ndarray | None = None
        self.alpha_mask: np.ndarray | None = None
        self.manual_mask: np.ndarray | None = None
        self.visual_marks: np.ndarray | None = None
        self.pending_visual_marks: np.ndarray | None = None

        self.display_geometry = DisplayGeometry()
        self.foreground_geometry = DisplayGeometry()
        self.mask_geometry = DisplayGeometry()
        self.tk_image: ImageTk.PhotoImage | None = None
        self.foreground_tk_image: ImageTk.PhotoImage | None = None
        self.mask_tk_image: ImageTk.PhotoImage | None = None
        self.operation_seconds = 0.0
        self.segmentation_count = 0
        self.mark_count = 0
        self.pending_edits = False
        self.last_saved_files: list[Path] = []
        self.mark_undo_stack: list[MarkUndoState] = []
        self.model_sessions: dict[str, object] = {}
        self.model_lock = threading.Lock()
        self.model_running = False
        self.main_zoom = 1.0
        self.main_pan_x = 0
        self.main_pan_y = 0
        self.last_pan_point: tuple[int, int] | None = None

        self.tool_var = tk.StringVar(value="brush")
        self.mark_mode_var = tk.StringVar(value="前景")
        self.zoom_label_var = tk.StringVar(value="100%")
        self.time_badge_var = tk.StringVar(value="0.0 s")
        self.interaction_badge_var = tk.StringVar(value="交互 0 次")
        self.canvas_state_var = tk.StringVar(value="等待操作")
        self.brush_size_var = tk.IntVar(value=15)
        self.iterations_var = tk.IntVar(value=1)
        self.use_opening_var = tk.BooleanVar(value=True)
        self.use_closing_var = tk.BooleanVar(value=True)
        self.keep_largest_var = tk.BooleanVar(value=False)
        self.smooth_edges_var = tk.BooleanVar(value=True)
        self.post_kernel_var = tk.IntVar(value=5)
        self.auto_expand_strokes_var = tk.BooleanVar(value=True)

        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.drag_started_at: float | None = None
        self.polygon_started_at: float | None = None
        self.action_grabcut_before: np.ndarray | None = None
        self.action_visual_before: np.ndarray | None = None
        self.action_result_before: np.ndarray | None = None
        self.action_alpha_before: np.ndarray | None = None
        self.action_manual_before: np.ndarray | None = None
        self.action_pending_before: np.ndarray | None = None
        self.action_pending_edits_before = False
        self.action_segmentation_count_before = 0
        self.action_mark_count_before = 0
        self.action_operation_seconds_before = 0.0
        self.last_brush_point: tuple[int, int] | None = None
        self.polygon_points: list[tuple[int, int]] = []

        self._build_ui()
        self._bind_canvas()
        self._load_default_sample()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = (UI_FONT_FAMILY, 9)
        title_font = (UI_FONT_FAMILY, 9, "bold")
        style.configure(".", font=base_font)
        style.configure("Sidebar.TFrame", background="#eef2f6")
        style.configure("Workspace.TFrame", background="#eef2f6")
        style.configure("Panel.TLabelframe", background="#eef2f6", bordercolor="#cbd5e1", relief="solid")
        style.configure("Panel.TLabelframe.Label", background="#eef2f6", foreground="#1f2937", font=title_font)
        style.configure("TLabel", background="#eef2f6", foreground="#1f2937")
        style.configure("Muted.TLabel", background="#eef2f6", foreground="#64748b")
        style.configure("TButton", padding=(8, 5), background="#f8fafc", foreground="#111827", bordercolor="#cbd5e1")
        style.map("TButton", background=[("active", "#e2e8f0")])
        style.configure("Primary.TButton", padding=(8, 7), background="#2563eb", foreground="#ffffff", font=title_font)
        style.map("Primary.TButton", background=[("active", "#1d4ed8")], foreground=[("active", "#ffffff")])
        style.configure("Danger.TButton", background="#fff1f2", foreground="#be123c")
        style.map("Danger.TButton", background=[("active", "#ffe4e6")])
        style.configure("TRadiobutton", background="#eef2f6", foreground="#1f2937")
        style.configure("TCheckbutton", background="#eef2f6", foreground="#1f2937")

    def _bubble_panel(self, parent: tk.Misc, padding: int | tuple[int, ...] = 12) -> tk.Frame:
        # Tkinter 不支持原生圆角卡片，这里用浅边框、留白和白色背景模拟轻盈气泡卡片。
        panel = tk.Frame(
            parent,
            bg="#ffffff",
            bd=0,
            highlightthickness=1,
            highlightbackground="#e8ecf3",
            highlightcolor="#dbe5f2",
            padx=padding if isinstance(padding, int) else 0,
            pady=padding if isinstance(padding, int) else 0,
        )
        if not isinstance(padding, int):
            left, top, right, bottom = padding
            panel.configure(padx=max(left, right), pady=max(top, bottom))
        return panel

    def _section_title(self, parent: tk.Misc, text: str, row: int, column: int = 0, columnspan: int = 1) -> None:
        tk.Label(
            parent,
            text=text,
            bg="#ffffff",
            fg="#263238",
            font=(UI_FONT_FAMILY, 10, "bold"),
        ).grid(row=row, column=column, columnspan=columnspan, sticky="w", pady=(0, 8))

    def _action_button(
        self,
        parent: tk.Misc,
        text: str,
        command,
        variant: str = "secondary",
    ) -> tk.Button:
        colors = {
            "primary": ("#5b7cfa", "#ffffff", "#4f6ff0"),
            "accent": ("#e8fbff", "#16798a", "#d8f6fc"),
            "secondary": ("#f3f6fb", "#263238", "#e8eef8"),
            "danger": ("#fff1f3", "#d6455d", "#ffe1e6"),
        }
        bg, fg, active_bg = colors.get(variant, colors["secondary"])
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=6,
            font=(UI_FONT_FAMILY, 9, "bold" if variant == "primary" else "normal"),
        )

    def _tool_toggle(self, parent: tk.Misc, text: str, value: str, variable: tk.StringVar) -> tk.Radiobutton:
        return tk.Radiobutton(
            parent,
            text=text,
            value=value,
            variable=variable,
            command=self._cancel_polygon,
            indicatoron=False,
            selectcolor="#eaf3ff",
            bg="#f3f6fb",
            fg="#263238",
            activebackground="#edf2ff",
            activeforeground="#263238",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=6,
            pady=5,
            font=(UI_FONT_FAMILY, 8),
        )

    def _badge(self, parent: tk.Misc, textvariable: tk.StringVar, bg: str = "#eef4ff", fg: str = "#4f63b5") -> tk.Label:
        return tk.Label(
            parent,
            textvariable=textvariable,
            bg=bg,
            fg=fg,
            padx=12,
            pady=6,
            font=(UI_FONT_FAMILY, 9, "bold"),
        )

    def _refresh_header_badges(self) -> None:
        self.time_badge_var.set(f"⏱ {self.operation_seconds:.1f} s")
        self.interaction_badge_var.set(f"✦ 交互 {self.segmentation_count} 次")
        if self.image_rgb is None:
            self.canvas_state_var.set("等待图片")
        elif self.model_running:
            self.canvas_state_var.set("模型处理中")
        elif self.result_mask is None:
            self.canvas_state_var.set("等待分割")
        elif self.pending_edits:
            self.canvas_state_var.set("待更新")
        else:
            self.canvas_state_var.set("已分割")

    def _build_ui(self) -> None:
        self._configure_style()
        self.root.configure(bg="#f5f7fb")
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)

        # 顶部 Header 使用一整块白色气泡卡片承载标题、文件操作和状态徽章。
        header = self._bubble_panel(self.root, padding=14)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 8))
        header.columnconfigure(1, weight=1)

        logo = tk.Canvas(header, width=46, height=46, bg="#ffffff", highlightthickness=0)
        logo.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 12))
        logo.create_oval(3, 3, 43, 43, fill="#edf2ff", outline="#dbe5ff")
        logo.create_text(23, 22, text="CV", fill="#5b7cfa", font=(UI_FONT_FAMILY, 11, "bold"))

        title_block = tk.Frame(header, bg="#ffffff")
        title_block.grid(row=0, column=1, rowspan=2, sticky="w")
        tk.Label(
            title_block,
            text="交互式图像分割",
            bg="#ffffff",
            fg="#263238",
            font=(UI_FONT_FAMILY, 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            title_block,
            text="Interactive Image Segmentation",
            bg="#ffffff",
            fg="#7a8493",
            font=(UI_FONT_FAMILY, 9),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        header_actions = tk.Frame(header, bg="#ffffff")
        header_actions.grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 10))
        self._action_button(header_actions, "打开图片", self.open_image, "primary").grid(row=0, column=0, padx=4)
        self._action_button(header_actions, "保存结果", self.save_results, "primary").grid(row=0, column=1, padx=4)
        self._action_button(header_actions, "保存透明图", self.save_transparent_foreground, "accent").grid(
            row=0, column=2, padx=4
        )

        header_badges = tk.Frame(header, bg="#ffffff")
        header_badges.grid(row=0, column=3, rowspan=2, sticky="e")
        self._badge(header_badges, self.time_badge_var, "#f1fffb", "#178f72").grid(row=0, column=0, padx=(0, 6))
        self._badge(header_badges, self.interaction_badge_var, "#f1fffb", "#178f72").grid(row=0, column=1)

        header_edit_actions = tk.Frame(header, bg="#ffffff")
        header_edit_actions.grid(row=0, column=4, rowspan=2, sticky="e", padx=(10, 0))
        self._action_button(header_edit_actions, "撤销", self.undo_last_mark, "secondary").grid(row=0, column=0, padx=4)
        self._action_button(header_edit_actions, "重置", self.reset_marks, "secondary").grid(row=0, column=1, padx=4)

        # 左侧工具区保持固定气泡卡片，尽量让所有常用功能在一个页面中展示。
        sidebar_shell = self._bubble_panel(self.root, padding=10)
        sidebar_shell.grid(row=1, column=0, sticky="ns", padx=(16, 8), pady=(8, 16))
        sidebar_shell.columnconfigure(0, weight=1)
        sidebar_shell.rowconfigure(0, weight=1)
        sidebar = tk.Frame(sidebar_shell, width=270, bg="#ffffff")
        sidebar.grid(row=0, column=0, sticky="new")
        sidebar.columnconfigure(0, weight=1)

        self._section_title(sidebar, "辅助功能", 0)
        action_grid = tk.Frame(sidebar, bg="#ffffff")
        action_grid.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        action_grid.columnconfigure(0, weight=1)
        action_grid.columnconfigure(1, weight=1)
        self._action_button(action_grid, "BiRefNet", self.model_matting_segment, "accent").grid(
            row=0, column=0, sticky="ew", padx=(0, 4), pady=3
        )
        self._action_button(action_grid, "胡须增强", self.enhance_whisker_details, "accent").grid(
            row=0, column=1, sticky="ew", padx=(4, 0), pady=3
        )
        self._action_button(action_grid, "边缘分割", self.edge_segment, "accent").grid(
            row=1, column=0, sticky="ew", padx=(0, 4), pady=(4, 0)
        )
        self._action_button(action_grid, "运行分割", self.segment, "primary").grid(
            row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0)
        )

        self._section_title(sidebar, "标记与参数", 4)
        view_panel = tk.Frame(sidebar, bg="#ffffff")
        view_panel.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        view_panel.columnconfigure(0, weight=1)
        view_panel.columnconfigure(1, weight=1)
        tk.Label(view_panel, text="标记类型", bg="#ffffff", fg="#263238").grid(row=0, column=0, sticky="w")
        ttk.OptionMenu(view_panel, self.mark_mode_var, self.mark_mode_var.get(), *MARK_MODES.keys()).grid(
            row=0, column=1, sticky="ew", padx=(8, 0), pady=1
        )

        split_panel = tk.Frame(view_panel, bg="#ffffff")
        split_panel.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        split_panel.columnconfigure(0, weight=1)
        split_panel.columnconfigure(1, weight=0)
        tk.Label(split_panel, text="画笔", bg="#ffffff", fg="#263238").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(split_panel, from_=3, to=60, textvariable=self.brush_size_var, width=6).grid(
            row=0, column=1, sticky="e", padx=(8, 0)
        )
        self._section_title(sidebar, "分割后优化", 6)
        optimize_panel = tk.Frame(sidebar, bg="#ffffff")
        optimize_panel.grid(row=7, column=0, sticky="ew", pady=(0, 8))
        optimize_panel.columnconfigure(0, weight=1)
        optimize_panel.columnconfigure(1, weight=1)
        ttk.Checkbutton(optimize_panel, text="开运算去噪", variable=self.use_opening_var).grid(row=0, column=0, sticky="w", pady=1)
        ttk.Checkbutton(optimize_panel, text="闭运算填洞", variable=self.use_closing_var).grid(row=0, column=1, sticky="w", pady=1)
        ttk.Checkbutton(optimize_panel, text="保留最大区域", variable=self.keep_largest_var).grid(row=1, column=0, sticky="w", pady=1)
        ttk.Checkbutton(optimize_panel, text="轮廓平滑", variable=self.smooth_edges_var).grid(row=1, column=1, sticky="w", pady=1)
        tk.Label(optimize_panel, text="优化强度", bg="#ffffff", fg="#263238").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Spinbox(optimize_panel, from_=3, to=15, increment=2, textvariable=self.post_kernel_var, width=8).grid(
            row=2, column=1, sticky="e", pady=(5, 0)
        )

        self._section_title(sidebar, "选择工具", 2)
        tool_grid = tk.Frame(sidebar, bg="#ffffff")
        tool_grid.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        tool_grid.columnconfigure(0, weight=1)
        tool_grid.columnconfigure(1, weight=1)
        tool_grid.columnconfigure(2, weight=1)
        tool_icons = {
            "brush": "● 涂抹",
            "eraser": "⌫ 橡皮擦",
            "line": "╱ 直线",
            "rectangle": "▭ 长方形",
            "square": "□ 正方形",
            "ellipse": "⬭ 椭圆",
            "circle": "○ 圆",
            "pentagon": "⬟ 五边形",
            "hexagon": "⬢ 六边形",
            "polygon": "◇ 任意多边形",
        }
        for index, tool in enumerate(TOOLS):
            self._tool_toggle(tool_grid, tool_icons[tool], tool, self.tool_var).grid(
                row=index // 3, column=index % 3, sticky="ew", padx=2, pady=2
        )

        zoom_bottom_panel = tk.Frame(sidebar_shell, bg="#ffffff")
        zoom_bottom_panel.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        zoom_bottom_panel.columnconfigure(0, weight=1)
        zoom_bottom_panel.columnconfigure(1, weight=1)
        zoom_bottom_panel.columnconfigure(2, weight=1)
        tk.Label(
            zoom_bottom_panel,
            textvariable=self.zoom_label_var,
            bg="#ffffff",
            fg="#7a8493",
            font=(UI_FONT_FAMILY, 9, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self._action_button(zoom_bottom_panel, "-", self.zoom_out_main).grid(
            row=1, column=0, sticky="ew", padx=(0, 3)
        )
        self._action_button(zoom_bottom_panel, "+", self.zoom_in_main).grid(
            row=1, column=1, sticky="ew", padx=3
        )
        self._action_button(zoom_bottom_panel, "适应", self.reset_main_zoom).grid(
            row=1, column=2, sticky="ew", padx=(3, 0)
        )

        self.status_var = tk.StringVar(value="准备就绪")

        canvas_frame = ttk.Frame(self.root, padding=(0, 8, 16, 16), style="Workspace.TFrame")
        canvas_frame.grid(row=1, column=1, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=3, uniform="views")
        canvas_frame.columnconfigure(1, weight=2, uniform="views")
        canvas_frame.rowconfigure(0, weight=1)

        result_frame = self._bubble_panel(canvas_frame, padding=14)
        result_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)
        result_header = tk.Frame(result_frame, bg="#ffffff")
        result_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        result_header.columnconfigure(0, weight=1)
        tk.Label(
            result_header,
            text="实验结果图像",
            bg="#ffffff",
            fg="#263238",
            font=(UI_FONT_FAMILY, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self._badge(result_header, self.canvas_state_var, "#edf2ff", "#5b7cfa").grid(row=0, column=1, sticky="e")

        right_frame = ttk.Frame(canvas_frame, style="Workspace.TFrame")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1, uniform="right_views")
        right_frame.rowconfigure(1, weight=1, uniform="right_views")

        foreground_frame = self._bubble_panel(right_frame, padding=12)
        foreground_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        foreground_frame.columnconfigure(0, weight=1)
        foreground_frame.rowconfigure(1, weight=1)
        tk.Label(
            foreground_frame,
            text="分割前景",
            bg="#ffffff",
            fg="#263238",
            font=(UI_FONT_FAMILY, 11, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        mask_frame = self._bubble_panel(right_frame, padding=12)
        mask_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        mask_frame.columnconfigure(0, weight=1)
        mask_frame.rowconfigure(1, weight=1)
        tk.Label(
            mask_frame,
            text="分割掩码",
            bg="#ffffff",
            fg="#263238",
            font=(UI_FONT_FAMILY, 11, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.canvas = tk.Canvas(result_frame, bg="#f2f5f9", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

        self.foreground_canvas = tk.Canvas(foreground_frame, bg="#f2f5f9", highlightthickness=0)
        self.foreground_canvas.grid(row=1, column=0, sticky="nsew")
        self.foreground_canvas.bind("<Configure>", lambda _event: self.redraw())

        self.mask_canvas = tk.Canvas(mask_frame, bg="#f2f5f9", highlightthickness=0)
        self.mask_canvas.grid(row=1, column=0, sticky="nsew")
        self.mask_canvas.bind("<Configure>", lambda _event: self.redraw())

    def _bind_canvas(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Shift-ButtonPress-1>", self.start_pan)
        self.canvas.bind("<Shift-B1-Motion>", self.pan_main)
        self.canvas.bind("<Shift-ButtonRelease-1>", self.end_pan)
        self.canvas.bind("<MouseWheel>", self.zoom_main_with_wheel)
        self.canvas.bind("<Double-Button-1>", self.finish_polygon)
        self.canvas.bind("<Button-3>", self.finish_polygon)
        self.root.bind_all("<Control-z>", self.undo_last_mark)
        self.root.bind_all("<Control-Z>", self.undo_last_mark)

    @staticmethod
    def _shift_pressed(event: tk.Event) -> bool:
        return bool(getattr(event, "state", 0) & 0x0001)

    def zoom_in_main(self) -> None:
        self._set_main_zoom(self.main_zoom * 1.25)

    def zoom_out_main(self) -> None:
        self._set_main_zoom(self.main_zoom / 1.25)

    def reset_main_zoom(self) -> None:
        self.main_zoom = 1.0
        self.main_pan_x = 0
        self.main_pan_y = 0
        self._update_zoom_label()
        self.redraw()

    def zoom_main_with_wheel(self, event: tk.Event) -> None:
        if self.image_rgb is None:
            return
        factor = 1.15 if getattr(event, "delta", 0) > 0 else 1 / 1.15
        self._set_main_zoom(self.main_zoom * factor, event.x, event.y)

    def start_pan(self, event: tk.Event) -> None:
        if self.image_rgb is None:
            return
        self.last_pan_point = (event.x, event.y)

    def pan_main(self, event: tk.Event) -> None:
        if self.image_rgb is None or self.last_pan_point is None:
            return
        last_x, last_y = self.last_pan_point
        self.main_pan_x += event.x - last_x
        self.main_pan_y += event.y - last_y
        self.last_pan_point = (event.x, event.y)
        self._clamp_main_pan()
        self.redraw()

    def end_pan(self, _event: tk.Event) -> None:
        self.last_pan_point = None

    def _set_main_zoom(self, zoom: float, center_x: int | None = None, center_y: int | None = None) -> None:
        if self.image_rgb is None:
            return
        old_geom = self.display_geometry
        old_zoom = self.main_zoom
        self.main_zoom = min(8.0, max(0.25, float(zoom)))
        if abs(self.main_zoom - old_zoom) < 0.001:
            return

        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_height, image_width = self.image_rgb.shape[:2]
        fit_scale = min(canvas_width / image_width, canvas_height / image_height)
        new_scale = fit_scale * self.main_zoom
        new_width = max(1, int(image_width * new_scale))
        new_height = max(1, int(image_height * new_scale))
        base_offset_x = (canvas_width - new_width) // 2
        base_offset_y = (canvas_height - new_height) // 2

        if center_x is not None and center_y is not None and old_geom.scale > 0:
            image_x = (center_x - old_geom.offset_x) / old_geom.scale
            image_y = (center_y - old_geom.offset_y) / old_geom.scale
            self.main_pan_x = int(round(center_x - image_x * new_scale - base_offset_x))
            self.main_pan_y = int(round(center_y - image_y * new_scale - base_offset_y))
        else:
            self.main_pan_x = int(round(self.main_pan_x * self.main_zoom / old_zoom))
            self.main_pan_y = int(round(self.main_pan_y * self.main_zoom / old_zoom))

        self._clamp_main_pan()
        self._update_zoom_label()
        self.redraw()

    def _update_zoom_label(self) -> None:
        self.zoom_label_var.set(f"{int(round(self.main_zoom * 100))}%")

    def _load_default_sample(self) -> None:
        default = Path("data") / "LENA.jpg"
        if default.exists():
            self.load_image(default)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图像",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")),
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        try:
            image = read_rgb_image(path)
        except Exception as exc:  # pragma: no cover - GUI 路径。
            messagebox.showerror("读取失败", str(exc))
            return

        height, width = image.shape[:2]
        self.image_path = path
        self.image_rgb = image
        self.grabcut_mask = np.full((height, width), GC_BGD, dtype=np.uint8)
        self.result_mask = None
        self.alpha_mask = None
        self.manual_mask = np.full((height, width), MANUAL_NONE, dtype=np.uint8)
        self.visual_marks = np.full((height, width), 255, dtype=np.uint8)
        self.pending_visual_marks = np.full((height, width), 255, dtype=np.uint8)
        self.segmentation_count = 0
        self.mark_count = 0
        self.pending_edits = False
        self.operation_seconds = 0.0
        self.main_zoom = 1.0
        self.main_pan_x = 0
        self.main_pan_y = 0
        self.last_pan_point = None
        self._update_zoom_label()
        self.drag_started_at = None
        self.polygon_started_at = None
        self.action_grabcut_before = None
        self.action_visual_before = None
        self.action_result_before = None
        self.action_alpha_before = None
        self.action_manual_before = None
        self.action_pending_before = None
        self.action_segmentation_count_before = 0
        self.action_mark_count_before = 0
        self.action_operation_seconds_before = 0.0
        self.mark_undo_stack.clear()
        self.polygon_points.clear()
        self.status_var.set("图片已准备")
        self.redraw()

    def reset_marks(self) -> None:
        if self.image_rgb is None:
            return
        height, width = self.image_rgb.shape[:2]
        self.grabcut_mask = np.full((height, width), GC_BGD, dtype=np.uint8)
        self.alpha_mask = None
        self.manual_mask = np.full((height, width), MANUAL_NONE, dtype=np.uint8)
        self.visual_marks = np.full((height, width), 255, dtype=np.uint8)
        self.pending_visual_marks = np.full((height, width), 255, dtype=np.uint8)
        self.result_mask = None
        self.segmentation_count = 0
        self.mark_count = 0
        self.pending_edits = False
        self.operation_seconds = 0.0
        self.drag_started_at = None
        self.polygon_started_at = None
        self.action_grabcut_before = None
        self.action_visual_before = None
        self.action_result_before = None
        self.action_alpha_before = None
        self.action_manual_before = None
        self.action_pending_before = None
        self.action_segmentation_count_before = 0
        self.action_mark_count_before = 0
        self.action_operation_seconds_before = 0.0
        self.mark_undo_stack.clear()
        self.polygon_points.clear()
        self.status_var.set("标记已重置")
        self.redraw()

    def segment(self) -> None:
        if self.image_rgb is None or self.grabcut_mask is None:
            messagebox.showinfo("提示", "请先读取图像")
            return

        started_at = time.monotonic()
        try:
            iterations = max(1, int(self.iterations_var.get()))
            expanded_initial = self._expand_initial_foreground_before_first_segment()
            self._apply_manual_mask_to_grabcut_mask()
            self.grabcut_mask, self.result_mask = run_grabcut(self.image_rgb, self.grabcut_mask, iterations)
            self._apply_manual_mask_to_grabcut_mask()
            self.result_mask = binary_from_grabcut_mask(self.grabcut_mask)
            self.result_mask = self._postprocess_result_mask(self.result_mask)
            self.alpha_mask = self.result_mask.copy()
        except Exception as exc:
            messagebox.showerror("分割失败", str(exc))
            return

        self.segmentation_count += 1
        self.pending_edits = False
        if self.pending_visual_marks is not None:
            self.pending_visual_marks.fill(255)
        self.mark_undo_stack.clear()
        self._record_operation_seconds(started_at)
        elapsed = self.operation_seconds
        extra = "\n已自动扩展初始涂抹区域" if expanded_initial else ""
        self.status_var.set(f"分割完成\n操作时间: {elapsed:.1f}s\n交互次数: {self.segmentation_count}{extra}")
        self.redraw()

    def edge_segment(self) -> None:
        if self.image_rgb is None or self.grabcut_mask is None:
            messagebox.showinfo("提示", "请先读取图像")
            return

        started_at = time.monotonic()
        try:
            edge_mask, canny_edges, seed_count = edge_watershed_segment(
                self.image_rgb,
                self.grabcut_mask,
                self.result_mask,
                self.manual_mask,
            )

            new_grabcut_mask = np.full(edge_mask.shape, GC_PR_BGD, dtype=np.uint8)
            new_grabcut_mask[edge_mask > 0] = GC_PR_FGD
            if self.manual_mask is not None:
                new_grabcut_mask[self.manual_mask == GC_FGD] = GC_FGD
                new_grabcut_mask[self.manual_mask == GC_PR_FGD] = GC_PR_FGD
                new_grabcut_mask[self.manual_mask == GC_BGD] = GC_BGD
                new_grabcut_mask[self.manual_mask == GC_PR_BGD] = GC_PR_BGD

            self.grabcut_mask = new_grabcut_mask
            self.result_mask = self._postprocess_result_mask(edge_mask)
            self.alpha_mask = self.result_mask.copy()
        except Exception as exc:
            messagebox.showerror("边缘分割失败", str(exc))
            return

        self.segmentation_count += 1
        self.pending_edits = False
        if self.pending_visual_marks is not None:
            self.pending_visual_marks.fill(255)
        self.mark_undo_stack.clear()
        self._record_operation_seconds(started_at)
        self.status_var.set(
            "边缘分割完成\n"
            f"Canny 边缘像素: {int(np.count_nonzero(canny_edges))}\n"
            f"前景种子区域: {seed_count}\n"
            f"操作时间: {self.operation_seconds:.1f}s\n"
            f"交互次数: {self.segmentation_count}"
        )
        self.redraw()

    def model_matting_segment(self) -> None:
        self._start_model_matting("birefnet-general", "BiRefNet")

    def _start_model_matting(self, model_name: str, display_name: str) -> None:
        if self.image_rgb is None or self.grabcut_mask is None:
            messagebox.showinfo("提示", "请先读取图像")
            return
        if self.model_running:
            messagebox.showinfo("提示", "模型正在处理中，请等待当前任务完成")
            return

        image_rgb = self.image_rgb.copy()
        started_at = time.monotonic()
        self.model_running = True
        self.status_var.set(f"{display_name} 正在处理...\nCPU 运行可能需要较久")
        self.redraw()

        # 模型推理较慢，放到后台线程，避免 Tkinter 主界面被系统判定为未响应。
        worker = threading.Thread(
            target=self._run_model_matting_worker,
            args=(image_rgb, started_at, model_name, display_name),
            daemon=True,
        )
        worker.start()

    def _run_model_matting_worker(
        self,
        image_rgb: np.ndarray,
        started_at: float,
        model_name: str,
        display_name: str,
    ) -> None:
        try:
            from rembg import new_session, remove
        except ImportError:
            self.root.after(
                0,
                self._finish_model_matting_error,
                display_name,
                "缺少依赖",
                '请先在 conda 环境中安装: python -m pip install "rembg[cpu]"',
            )
            return

        try:
            with self.model_lock:
                session = self.model_sessions.get(model_name)
                if session is None:
                    session = new_session(model_name)
                    self.model_sessions[model_name] = session
            input_image = Image.fromarray(image_rgb, mode="RGB")
            try:
                output_image = remove(input_image, session=session, putalpha=True).convert("RGBA")
            except TypeError:
                output_image = remove(input_image, session=session).convert("RGBA")
            alpha = np.array(output_image, dtype=np.uint8)[:, :, 3]
        except Exception as exc:
            self.root.after(0, self._finish_model_matting_error, display_name, f"{display_name} 失败", str(exc))
            return

        self.root.after(0, self._finish_model_matting_success, alpha, started_at, display_name)

    def _finish_model_matting_error(self, display_name: str, title: str, message: str) -> None:
        self.model_running = False
        self.status_var.set(f"{display_name} 处理失败")
        self.redraw()
        messagebox.showerror(title, message)

    def _finish_model_matting_success(self, alpha: np.ndarray, started_at: float, display_name: str) -> None:
        if self.image_rgb is None or self.grabcut_mask is None:
            self.model_running = False
            self.redraw()
            return
        if alpha.shape != self.grabcut_mask.shape:
            self.model_running = False
            self.status_var.set(f"{display_name} 结果尺寸与当前图像不一致")
            self.redraw()
            messagebox.showerror(f"{display_name} 失败", "模型结果尺寸与当前图像不一致，请重新读取图像后再试")
            return

        foreground = alpha >= 64
        strong_foreground = alpha >= 230
        sure_background = alpha <= 12

        self.alpha_mask = alpha
        self.result_mask = np.where(foreground, 255, 0).astype(np.uint8)
        self.grabcut_mask = np.full(alpha.shape, GC_PR_BGD, dtype=np.uint8)
        self.grabcut_mask[foreground] = GC_PR_FGD
        self.grabcut_mask[strong_foreground] = GC_FGD
        self.grabcut_mask[sure_background] = GC_BGD

        if self.manual_mask is not None:
            self.manual_mask.fill(MANUAL_NONE)
        if self.visual_marks is not None:
            self.visual_marks.fill(255)
        if self.pending_visual_marks is not None:
            self.pending_visual_marks.fill(255)

        self.pending_edits = False
        self.segmentation_count += 1
        self.mark_undo_stack.clear()
        self.model_running = False
        self._record_operation_seconds(started_at)
        self.status_var.set(
            f"{display_name} 完成\n"
            f"操作时间: {self.operation_seconds:.1f}s\n"
            f"交互次数: {self.segmentation_count}"
        )
        self.redraw()

    def enhance_whisker_details(self) -> None:
        if self.image_rgb is None or self.grabcut_mask is None:
            messagebox.showinfo("提示", "请先读取图像")
            return
        if self.result_mask is None:
            messagebox.showinfo("提示", "请先运行一次分割或模型抠图")
            return
        if self.manual_mask is None:
            messagebox.showinfo("提示", "请先用前景笔刷在胡须附近画几笔")
            return

        seed_mask = (self.manual_mask == GC_FGD) | (self.manual_mask == GC_PR_FGD)
        if not np.any(seed_mask):
            messagebox.showinfo("提示", "请先用前景笔刷在胡须附近画几笔")
            return

        blocked_mask = (self.manual_mask == GC_BGD) | (self.manual_mask == GC_PR_BGD)
        started_at = time.monotonic()
        self._snapshot_action_masks()
        try:
            enhanced_mask, enhanced_alpha, detail_mask, added_pixels = enhance_bright_line_details(
                self.image_rgb,
                self.result_mask,
                seed_mask,
                self.alpha_mask,
                blocked_mask=blocked_mask,
            )
        except Exception as exc:
            self._clear_action_snapshot()
            messagebox.showerror("胡须增强失败", str(exc))
            return

        if added_pixels <= 0:
            self._clear_action_snapshot()
            self.status_var.set("胡须增强未找到可加入的细线\n请把前景笔刷画在胡须附近")
            return

        detail_area = detail_mask > 0
        self.result_mask = enhanced_mask
        self.alpha_mask = enhanced_alpha
        self.grabcut_mask[detail_area] = GC_FGD
        self.manual_mask[detail_area] = GC_FGD
        if self.visual_marks is not None:
            self.visual_marks[detail_area] = GC_FGD
        if self.pending_visual_marks is not None:
            self.pending_visual_marks.fill(255)

        self.pending_edits = False
        self.segmentation_count += 1
        self.mark_count += 1
        if self._action_masks_changed():
            self._push_mark_undo_from_action_snapshot()
        self._record_operation_seconds(started_at)
        self._clear_action_snapshot()
        self.status_var.set(
            f"胡须增强完成\n"
            f"新增细节像素: {added_pixels}\n"
            f"操作时间: {self.operation_seconds:.1f}s\n"
            f"交互次数: {self.segmentation_count}"
        )
        self.redraw()

    def save_results(self) -> None:
        if self.image_rgb is None or self.image_path is None:
            messagebox.showinfo("提示", "请先读取图像")
            return
        if self.result_mask is None or self.pending_edits:
            self.segment()
            if self.result_mask is None:
                return

        target_dir = filedialog.askdirectory(title="选择保存目录", initialdir=str(self.image_path.parent))
        if not target_dir:
            return

        stem = self.image_path.stem
        date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(target_dir)
        overlay_path = out_dir / f"{stem}_overlay_{date_suffix}.png"
        mask_path = out_dir / f"{stem}_mask_{date_suffix}.png"
        foreground_path = out_dir / f"{stem}_foreground_{date_suffix}.png"
        alpha_path = out_dir / f"{stem}_alpha_{date_suffix}.png"

        overlay = self.build_view_image("叠加结果", annotate=True)
        foreground = self.build_foreground_image(annotate=False)
        Image.fromarray(self.current_binary_mask(), mode="L").save(mask_path)
        write_rgb_image(overlay_path, overlay)
        write_rgb_image(foreground_path, foreground)
        saved_files = [overlay_path, mask_path, foreground_path]
        if self.alpha_mask is not None:
            Image.fromarray(self.alpha_mask, mode="L").save(alpha_path)
            saved_files.append(alpha_path)

        self.last_saved_files = saved_files
        self.status_var.set("保存完成:\n" + "\n".join(path.name for path in self.last_saved_files))
        messagebox.showinfo("保存完成", "\n".join(str(path) for path in self.last_saved_files))

    def save_transparent_foreground(self) -> None:
        if self.image_rgb is None or self.image_path is None:
            messagebox.showinfo("提示", "请先读取图像")
            return
        if self.result_mask is None or self.pending_edits:
            self.segment()
            if self.result_mask is None:
                return

        default_dir = TASK1_DIR.parent / "task3" / "foreground"
        if not default_dir.exists():
            default_dir.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem
        date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = default_dir / f"{stem}_transparent_{date_suffix}.png"

        transparent = self.build_transparent_foreground_image()
        Image.fromarray(transparent, mode="RGBA").save(saved_path)
        self.last_saved_files = [saved_path]
        self.status_var.set(f"透明背景图已保存:\n{saved_path}")
        messagebox.showinfo("保存完成", str(saved_path))

    def on_press(self, event: tk.Event) -> None:
        if self._shift_pressed(event):
            return
        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return

        if self.tool_var.get() == "polygon":
            if not self.polygon_points:
                self.polygon_started_at = time.monotonic()
                self._snapshot_action_masks()
            self.polygon_points.append(point)
            self.drag_current = point
            self.redraw()
            return

        self.drag_start = point
        self.drag_current = point
        self.drag_started_at = time.monotonic()
        self._snapshot_action_masks()
        self.last_brush_point = point
        if self.tool_var.get() in ("brush", "eraser"):
            self._draw_brush_segment(point, point)
            self.redraw()

    def on_drag(self, event: tk.Event) -> None:
        if self._shift_pressed(event):
            return
        point = (
            self.canvas_to_image(event.x, event.y)
            if self.tool_var.get() in ("brush", "eraser")
            else self.canvas_to_image_clamped(event.x, event.y)
        )
        if point is None:
            return

        if self.tool_var.get() == "polygon":
            self.drag_current = point
            self.redraw()
            return

        if self.tool_var.get() in ("brush", "eraser"):
            if self.last_brush_point is not None:
                self._draw_brush_segment(self.last_brush_point, point)
                self.last_brush_point = point
                self.redraw()
            return

        self.drag_current = point
        self.redraw()

    def on_release(self, event: tk.Event) -> None:
        if self._shift_pressed(event):
            return
        point = (
            self.canvas_to_image(event.x, event.y)
            if self.tool_var.get() in ("polygon", "brush", "eraser")
            else self.canvas_to_image_clamped(event.x, event.y)
        )
        if point is not None:
            self.drag_current = point

        if self.tool_var.get() == "polygon":
            self.redraw()
            return

        if self.tool_var.get() in ("brush", "eraser"):
            if (
                self.tool_var.get() in ("brush", "eraser")
                and self.drag_start is not None
                and self._action_masks_changed()
            ):
                self._push_mark_undo_from_action_snapshot()
                self._record_operation_seconds(self.drag_started_at)
                self.mark_count += 1
            self.drag_start = None
            self.drag_current = None
            self.drag_started_at = None
            self._clear_action_snapshot()
            self.last_brush_point = None
            self.redraw()
            return

        if (
            self.drag_start is not None
            and self.drag_current is not None
            and self._drag_distance(self.drag_start, self.drag_current) >= 2.0
        ):
            self._commit_shape(self.tool_var.get(), self.drag_start, self.drag_current)
            if self._action_masks_changed():
                self._push_mark_undo_from_action_snapshot()
                self._record_operation_seconds(self.drag_started_at)
                self.mark_count += 1

        self.drag_start = None
        self.drag_current = None
        self.drag_started_at = None
        self._clear_action_snapshot()
        self.redraw()

    def finish_polygon(self, _event: tk.Event | None = None) -> None:
        if self.tool_var.get() != "polygon" or len(self.polygon_points) < 3:
            return
        self._commit_polygon(self.polygon_points)
        if self._action_masks_changed():
            self._push_mark_undo_from_action_snapshot()
            self._record_operation_seconds(self.polygon_started_at)
            self.mark_count += 1
        self.polygon_points.clear()
        self.polygon_started_at = None
        self._clear_action_snapshot()
        self.drag_current = None
        self.redraw()

    def _cancel_polygon(self) -> None:
        self.polygon_points.clear()
        self.polygon_started_at = None
        self._clear_action_snapshot()
        self.drag_current = None
        self.redraw()

    def _current_mark_value(self) -> int:
        return MARK_MODES.get(self.mark_mode_var.get(), GC_FGD)

    def _prepare_initial_region(self) -> bool:
        return self.mark_mode_var.get() == "初始区域" and not self._manual_priority_active()

    def _manual_priority_active(self) -> bool:
        return self.segmentation_count > 0 or self.result_mask is not None

    def _apply_manual_mask_to_grabcut_mask(self) -> None:
        if self.grabcut_mask is None or self.manual_mask is None:
            return
        manual_area = self.manual_mask != MANUAL_NONE
        self.grabcut_mask[manual_area] = self.manual_mask[manual_area]

    def _apply_manual_mask_to_result_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.manual_mask is None:
            return mask
        output = mask.copy()
        output[(self.manual_mask == GC_FGD) | (self.manual_mask == GC_PR_FGD)] = 255
        output[(self.manual_mask == GC_BGD) | (self.manual_mask == GC_PR_BGD)] = 0
        return output

    def _expand_initial_foreground_before_first_segment(self) -> bool:
        if (
            self.grabcut_mask is None
            or not self.auto_expand_strokes_var.get()
            or self.segmentation_count > 0
            or self.result_mask is not None
        ):
            return False
        self.grabcut_mask, expanded = expand_initial_foreground_from_marks(
            self.grabcut_mask,
            self.visual_marks,
        )
        return expanded

    def _mark_mask_edited(self) -> None:
        if self.result_mask is not None:
            self.pending_edits = True
            self.status_var.set("标记已修改\n点击运行分割更新前景")

    def _postprocess_result_mask(self, mask: np.ndarray) -> np.ndarray:
        processed = postprocess_binary_mask(
            mask,
            use_opening=self.use_opening_var.get(),
            use_closing=self.use_closing_var.get(),
            keep_largest=self.keep_largest_var.get(),
            smooth_edges=self.smooth_edges_var.get(),
            kernel_size=self.post_kernel_var.get(),
        )
        if self.grabcut_mask is not None:
            processed[self.grabcut_mask == GC_FGD] = 255
            processed[self.grabcut_mask == GC_BGD] = 0
        processed = self._apply_manual_mask_to_result_mask(processed)
        return processed

    def _record_operation_seconds(self, started_at: float | None) -> None:
        if started_at is None:
            return
        self.operation_seconds += max(0.0, time.monotonic() - started_at)

    def _snapshot_action_masks(self) -> None:
        self.action_grabcut_before = None if self.grabcut_mask is None else self.grabcut_mask.copy()
        self.action_visual_before = None if self.visual_marks is None else self.visual_marks.copy()
        self.action_result_before = None if self.result_mask is None else self.result_mask.copy()
        self.action_alpha_before = None if self.alpha_mask is None else self.alpha_mask.copy()
        self.action_manual_before = None if self.manual_mask is None else self.manual_mask.copy()
        self.action_pending_before = None if self.pending_visual_marks is None else self.pending_visual_marks.copy()
        self.action_pending_edits_before = getattr(self, "pending_edits", False)
        self.action_segmentation_count_before = getattr(self, "segmentation_count", 0)
        self.action_mark_count_before = getattr(self, "mark_count", 0)
        self.action_operation_seconds_before = getattr(self, "operation_seconds", 0.0)

    def _action_masks_changed(self) -> bool:
        changed = False
        if self.action_grabcut_before is not None and self.grabcut_mask is not None:
            changed = changed or not np.array_equal(self.action_grabcut_before, self.grabcut_mask)
        if self.action_visual_before is not None and self.visual_marks is not None:
            changed = changed or not np.array_equal(self.action_visual_before, self.visual_marks)
        if self.action_result_before is not None and self.result_mask is not None:
            changed = changed or not np.array_equal(self.action_result_before, self.result_mask)
        if self.action_alpha_before is not None and self.alpha_mask is not None:
            changed = changed or not np.array_equal(self.action_alpha_before, self.alpha_mask)
        if self.action_manual_before is not None and self.manual_mask is not None:
            changed = changed or not np.array_equal(self.action_manual_before, self.manual_mask)
        if self.action_pending_before is not None and self.pending_visual_marks is not None:
            changed = changed or not np.array_equal(self.action_pending_before, self.pending_visual_marks)
        return changed

    def _clear_action_snapshot(self) -> None:
        self.action_grabcut_before = None
        self.action_visual_before = None
        self.action_result_before = None
        self.action_alpha_before = None
        self.action_manual_before = None
        self.action_pending_before = None
        self.action_pending_edits_before = False
        self.action_segmentation_count_before = 0
        self.action_mark_count_before = 0
        self.action_operation_seconds_before = 0.0

    def _push_mark_undo_from_action_snapshot(self) -> None:
        self.mark_undo_stack.append(
            MarkUndoState(
                grabcut_mask=None if self.action_grabcut_before is None else self.action_grabcut_before.copy(),
                result_mask=None if self.action_result_before is None else self.action_result_before.copy(),
                alpha_mask=None if self.action_alpha_before is None else self.action_alpha_before.copy(),
                manual_mask=None if self.action_manual_before is None else self.action_manual_before.copy(),
                visual_marks=None if self.action_visual_before is None else self.action_visual_before.copy(),
                pending_visual_marks=None
                if self.action_pending_before is None
                else self.action_pending_before.copy(),
                pending_edits=self.action_pending_edits_before,
                segmentation_count=self.action_segmentation_count_before,
                mark_count=self.action_mark_count_before,
                operation_seconds=self.action_operation_seconds_before,
            )
        )
        if len(self.mark_undo_stack) > 30:
            del self.mark_undo_stack[0]

    def undo_last_mark(self, _event: tk.Event | None = None) -> None:
        if not self.mark_undo_stack:
            self.status_var.set("没有可撤销的标记")
            return

        state = self.mark_undo_stack.pop()
        self.grabcut_mask = None if state.grabcut_mask is None else state.grabcut_mask.copy()
        self.result_mask = None if state.result_mask is None else state.result_mask.copy()
        self.alpha_mask = None if state.alpha_mask is None else state.alpha_mask.copy()
        self.manual_mask = None if state.manual_mask is None else state.manual_mask.copy()
        self.visual_marks = None if state.visual_marks is None else state.visual_marks.copy()
        self.pending_visual_marks = (
            None if state.pending_visual_marks is None else state.pending_visual_marks.copy()
        )
        self.pending_edits = state.pending_edits
        self.segmentation_count = state.segmentation_count
        self.mark_count = state.mark_count
        self.operation_seconds = state.operation_seconds
        self.status_var.set("已撤销上一次标记")
        self.redraw()

    def _mark_layers_for_current_action(self) -> list[np.ndarray]:
        layers = []
        if self.visual_marks is not None:
            layers.append(self.visual_marks)
        if self.result_mask is not None and self.pending_visual_marks is not None:
            layers.append(self.pending_visual_marks)
        return layers

    @staticmethod
    def _drag_distance(start: tuple[int, int], end: tuple[int, int]) -> float:
        return math.hypot(end[0] - start[0], end[1] - start[1])

    def _mask_value_for_tool(self, tool: str) -> int:
        value = self._current_mark_value()
        if self._manual_priority_active():
            return value
        if tool not in ("brush", "eraser", "line") and value == GC_FGD:
            return GC_PR_FGD
        if tool not in ("brush", "eraser", "line") and value == GC_PR_BGD:
            return GC_BGD
        return value

    def _draw_brush_segment(self, p1: tuple[int, int], p2: tuple[int, int]) -> None:
        if self.grabcut_mask is None or self.visual_marks is None:
            return
        cv2 = require_cv2()
        thickness = max(1, int(self.brush_size_var.get()))
        if self.tool_var.get() == "eraser":
            cv2.line(self.grabcut_mask, p1, p2, GC_BGD, thickness)
            if self._manual_priority_active() and self.manual_mask is not None:
                cv2.line(self.manual_mask, p1, p2, GC_BGD, thickness)
            if self.result_mask is not None:
                cv2.line(self.result_mask, p1, p2, 0, thickness)
            if self.alpha_mask is not None:
                cv2.line(self.alpha_mask, p1, p2, 0, thickness)
            cv2.line(self.visual_marks, p1, p2, 255, thickness)
            if self.pending_visual_marks is not None:
                cv2.line(self.pending_visual_marks, p1, p2, 255, thickness)
        else:
            value = self._mask_value_for_tool("brush")
            cv2.line(self.grabcut_mask, p1, p2, value, thickness)
            if self._manual_priority_active() and self.manual_mask is not None:
                cv2.line(self.manual_mask, p1, p2, value, thickness)
            cv2.line(self.visual_marks, p1, p2, value, thickness)
            if self.result_mask is not None and self.pending_visual_marks is not None:
                cv2.line(self.pending_visual_marks, p1, p2, value, thickness)
        self._mark_mask_edited()

    def _commit_shape(self, tool: str, start: tuple[int, int], end: tuple[int, int]) -> None:
        if self.grabcut_mask is None or self.visual_marks is None:
            return
        if self._prepare_initial_region():
            self.grabcut_mask.fill(GC_BGD)
            self.visual_marks.fill(255)
            if self.pending_visual_marks is not None:
                self.pending_visual_marks.fill(255)

        points = self._shape_points(tool, start, end)
        value = self._mask_value_for_tool(tool)
        thickness = max(1, int(self.brush_size_var.get()))
        self._draw_shape_on_masks(tool, start, end, points, value, thickness)
        self._mark_mask_edited()

    def _commit_polygon(self, points: Iterable[tuple[int, int]]) -> None:
        if self.grabcut_mask is None or self.visual_marks is None:
            return
        cv2 = require_cv2()
        if self._prepare_initial_region():
            self.grabcut_mask.fill(GC_BGD)
            self.visual_marks.fill(255)
            if self.pending_visual_marks is not None:
                self.pending_visual_marks.fill(255)
        value = self._mask_value_for_tool("polygon")
        pts = np.array(list(points), dtype=np.int32)
        cv2.fillPoly(self.grabcut_mask, [pts], value)
        if self._manual_priority_active() and self.manual_mask is not None:
            cv2.fillPoly(self.manual_mask, [pts], value)
        for marks in self._mark_layers_for_current_action():
            if self._prepare_initial_region():
                cv2.polylines(marks, [pts], True, value, max(3, int(self.brush_size_var.get())))
            else:
                cv2.polylines(marks, [pts], True, value, max(2, int(self.brush_size_var.get())))
                cv2.fillPoly(marks, [pts], value)
        self._mark_mask_edited()

    def _draw_shape_on_masks(
        self,
        tool: str,
        start: tuple[int, int],
        end: tuple[int, int],
        points: list[tuple[int, int]],
        value: int,
        thickness: int,
    ) -> None:
        cv2 = require_cv2()
        assert self.grabcut_mask is not None
        assert self.visual_marks is not None
        mark_layers = self._mark_layers_for_current_action()
        manual_layer = self.manual_mask if self._manual_priority_active() else None

        if tool == "line":
            cv2.line(self.grabcut_mask, start, end, value, thickness)
            if manual_layer is not None:
                cv2.line(manual_layer, start, end, value, thickness)
            for marks in mark_layers:
                cv2.line(marks, start, end, value, thickness)
            return

        if tool in ("rectangle", "square"):
            x1, y1, x2, y2 = self._bbox_from_points(start, end, square=(tool == "square"))
            cv2.rectangle(self.grabcut_mask, (x1, y1), (x2, y2), value, -1)
            if manual_layer is not None:
                cv2.rectangle(manual_layer, (x1, y1), (x2, y2), value, -1)
            visual_thickness = 3 if self._prepare_initial_region() else -1
            for marks in mark_layers:
                cv2.rectangle(marks, (x1, y1), (x2, y2), value, visual_thickness)
            return

        if tool in ("ellipse", "circle"):
            x1, y1, x2, y2 = self._bbox_from_points(start, end, square=(tool == "circle"))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, abs(x2 - x1) // 2), max(1, abs(y2 - y1) // 2))
            cv2.ellipse(self.grabcut_mask, center, axes, 0, 0, 360, value, -1)
            if manual_layer is not None:
                cv2.ellipse(manual_layer, center, axes, 0, 0, 360, value, -1)
            visual_thickness = 3 if self._prepare_initial_region() else -1
            for marks in mark_layers:
                cv2.ellipse(marks, center, axes, 0, 0, 360, value, visual_thickness)
            return

        if points:
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(self.grabcut_mask, [pts], value)
            if manual_layer is not None:
                cv2.fillPoly(manual_layer, [pts], value)
            for marks in mark_layers:
                if self._prepare_initial_region():
                    cv2.polylines(marks, [pts], True, value, 3)
                else:
                    cv2.fillPoly(marks, [pts], value)

    def _shape_points(self, tool: str, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        if tool == "pentagon":
            return self._regular_polygon_points(start, end, sides=5)
        if tool == "hexagon":
            return self._regular_polygon_points(start, end, sides=6)
        return []

    def _regular_polygon_points(
        self, start: tuple[int, int], end: tuple[int, int], sides: int
    ) -> list[tuple[int, int]]:
        cx = (start[0] + end[0]) / 2
        cy = (start[1] + end[1]) / 2
        rx = abs(end[0] - start[0]) / 2
        ry = abs(end[1] - start[1]) / 2
        radius = max(1.0, min(rx, ry) if rx and ry else max(rx, ry))
        return [
            (
                int(round(cx + radius * math.cos(-math.pi / 2 + 2 * math.pi * i / sides))),
                int(round(cy + radius * math.sin(-math.pi / 2 + 2 * math.pi * i / sides))),
            )
            for i in range(sides)
        ]

    def _bbox_from_points(
        self, start: tuple[int, int], end: tuple[int, int], square: bool = False
    ) -> tuple[int, int, int, int]:
        x1, y1 = start
        x2, y2 = end
        if square:
            side = max(abs(x2 - x1), abs(y2 - y1))
            x2 = x1 + side * (1 if x2 >= x1 else -1)
            y2 = y1 + side * (1 if y2 >= y1 else -1)
        if self.image_rgb is not None:
            height, width = self.image_rgb.shape[:2]
            x1, x2 = np.clip([x1, x2], 0, width - 1)
            y1, y2 = np.clip([y1, y2], 0, height - 1)
        return int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))

    def canvas_to_image(self, x: int, y: int) -> tuple[int, int] | None:
        if self.image_rgb is None:
            return None
        geom = self.display_geometry
        ix = int((x - geom.offset_x) / geom.scale)
        iy = int((y - geom.offset_y) / geom.scale)
        height, width = self.image_rgb.shape[:2]
        if ix < 0 or iy < 0 or ix >= width or iy >= height:
            return None
        return ix, iy

    def canvas_to_image_clamped(self, x: int, y: int) -> tuple[int, int] | None:
        if self.image_rgb is None:
            return None
        geom = self.display_geometry
        ix = int((x - geom.offset_x) / geom.scale)
        iy = int((y - geom.offset_y) / geom.scale)
        height, width = self.image_rgb.shape[:2]
        ix = int(np.clip(ix, 0, width - 1))
        iy = int(np.clip(iy, 0, height - 1))
        return ix, iy

    def image_to_canvas(self, point: tuple[int, int]) -> tuple[int, int]:
        geom = self.display_geometry
        return int(point[0] * geom.scale + geom.offset_x), int(point[1] * geom.scale + geom.offset_y)

    def current_binary_mask(self) -> np.ndarray:
        if self.image_rgb is None:
            return np.zeros((480, 640), dtype=np.uint8)
        if self.result_mask is not None:
            return self.result_mask
        if self.grabcut_mask is not None:
            return binary_from_grabcut_mask(self.grabcut_mask)
        height, width = self.image_rgb.shape[:2]
        return np.zeros((height, width), dtype=np.uint8)

    def current_alpha_mask(self) -> np.ndarray:
        if self.image_rgb is None:
            return np.zeros((480, 640), dtype=np.uint8)
        if self.alpha_mask is not None:
            return self.alpha_mask
        return self.current_binary_mask()

    def build_view_image(self, view: str, annotate: bool = True) -> np.ndarray:
        if self.image_rgb is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        base = self.image_rgb.copy()
        if view == "原图":
            output = base
        elif view == "前景图":
            mask = self.current_binary_mask()
            output = (base * (mask[:, :, None] > 0)).astype(np.uint8)
        else:
            output = self._overlay_result(base)

        if annotate:
            output = self._annotate(output)
        return output

    def build_mask_image(self, annotate: bool = True) -> np.ndarray:
        mask = self.current_alpha_mask()
        output = np.repeat(mask[:, :, None], 3, axis=2)
        if annotate:
            output = self._annotate(output)
        return output

    def build_foreground_image(self, annotate: bool = True) -> np.ndarray:
        if self.image_rgb is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        alpha = self.current_alpha_mask().astype(np.float32) / 255.0
        output = (self.image_rgb.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
        if annotate:
            output = self._annotate(output)
        return output

    def build_transparent_foreground_image(self) -> np.ndarray:
        if self.image_rgb is None:
            return np.zeros((480, 640, 4), dtype=np.uint8)
        alpha = self.current_alpha_mask().astype(np.uint8)
        rgb = bleed_edge_colors(self.image_rgb, alpha)
        return np.dstack((rgb, alpha)).astype(np.uint8)

    def _overlay_result(self, base: np.ndarray) -> np.ndarray:
        cv2 = require_cv2()
        output = base.copy()

        if self.result_mask is not None:
            green = np.zeros_like(output)
            green[:, :, 1] = 210
            foreground = self.result_mask > 0
            output[foreground] = (output[foreground] * 0.62 + green[foreground] * 0.38).astype(np.uint8)
            contours, _ = cv2.findContours(self.result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (255, 230, 0), 2)

        if self.visual_marks is not None and self.result_mask is None:
            marks = self.visual_marks
            fg = (marks == GC_FGD) | (marks == GC_PR_FGD)
            bg = (marks == GC_BGD) | (marks == GC_PR_BGD)
            green = np.array([0, 255, 80], dtype=np.uint8)
            red = np.array([255, 60, 50], dtype=np.uint8)
            output[fg] = (output[fg] * 0.45 + green * 0.55).astype(np.uint8)
            output[bg] = (output[bg] * 0.45 + red * 0.55).astype(np.uint8)
        elif self.pending_visual_marks is not None and self.pending_edits:
            marks = self.pending_visual_marks
            fg = (marks == GC_FGD) | (marks == GC_PR_FGD)
            bg = (marks == GC_BGD) | (marks == GC_PR_BGD)
            cyan = np.array([0, 210, 255], dtype=np.uint8)
            red = np.array([255, 60, 50], dtype=np.uint8)
            output[fg] = (output[fg] * 0.35 + cyan * 0.65).astype(np.uint8)
            output[bg] = (output[bg] * 0.35 + red * 0.65).astype(np.uint8)

        return output

    def _annotate(self, image_rgb: np.ndarray) -> np.ndarray:
        image = Image.fromarray(image_rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        elapsed = self.operation_seconds
        text = f"Time: {elapsed:.1f}s   Interactions: {self.segmentation_count}"
        font = load_annotation_font(16)
        bbox = draw.textbbox((0, 0), text, font=font)
        x, y = 12, 12
        draw.rectangle((x - 6, y - 5, x + bbox[2] + 6, y + bbox[3] + 6), fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        return np.array(image)

    def redraw(self) -> None:
        self._refresh_header_badges()
        self._draw_result_canvas()
        self._draw_foreground_canvas()
        self._draw_mask_canvas()

    def _draw_result_canvas(self) -> None:
        self.canvas.delete("all")
        if self.image_rgb is None:
            self._draw_empty_message(self.canvas, "请读取图像")
            return

        display = Image.fromarray(self.build_view_image("叠加结果", annotate=True), mode="RGB")
        self.display_geometry, self.tk_image = self._paint_main_image_on_canvas(display)
        self._draw_preview()

    def _draw_foreground_canvas(self) -> None:
        self.foreground_canvas.delete("all")
        if self.image_rgb is None:
            self._draw_empty_message(self.foreground_canvas, "暂无前景")
            return

        display = Image.fromarray(self.build_foreground_image(annotate=True), mode="RGB")
        self.foreground_geometry, self.foreground_tk_image = self._paint_image_on_canvas(
            self.foreground_canvas, display
        )

    def _draw_mask_canvas(self) -> None:
        self.mask_canvas.delete("all")
        if self.image_rgb is None:
            self._draw_empty_message(self.mask_canvas, "暂无掩码")
            return

        display = Image.fromarray(self.build_mask_image(annotate=True), mode="RGB")
        self.mask_geometry, self.mask_tk_image = self._paint_image_on_canvas(self.mask_canvas, display)

    def _paint_image_on_canvas(
        self, canvas: tk.Canvas, image: Image.Image
    ) -> tuple[DisplayGeometry, ImageTk.PhotoImage]:
        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())
        scale = min(canvas_width / image.width, canvas_height / image.height)
        new_width = max(1, int(image.width * scale))
        new_height = max(1, int(image.height * scale))
        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized)
        canvas.create_image(offset_x, offset_y, image=tk_image, anchor="nw")
        return DisplayGeometry(scale, offset_x, offset_y, new_width, new_height), tk_image

    def _paint_main_image_on_canvas(self, image: Image.Image) -> tuple[DisplayGeometry, ImageTk.PhotoImage]:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        fit_scale = min(canvas_width / image.width, canvas_height / image.height)
        scale = fit_scale * self.main_zoom
        new_width = max(1, int(image.width * scale))
        new_height = max(1, int(image.height * scale))
        base_offset_x = (canvas_width - new_width) // 2
        base_offset_y = (canvas_height - new_height) // 2
        self._clamp_main_pan_for_size(canvas_width, canvas_height, new_width, new_height, base_offset_x, base_offset_y)
        offset_x = base_offset_x + self.main_pan_x
        offset_y = base_offset_y + self.main_pan_y
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(offset_x, offset_y, image=tk_image, anchor="nw")
        return DisplayGeometry(scale, offset_x, offset_y, new_width, new_height), tk_image

    def _clamp_main_pan(self) -> None:
        if self.image_rgb is None:
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image_height, image_width = self.image_rgb.shape[:2]
        fit_scale = min(canvas_width / image_width, canvas_height / image_height)
        scale = fit_scale * self.main_zoom
        new_width = max(1, int(image_width * scale))
        new_height = max(1, int(image_height * scale))
        base_offset_x = (canvas_width - new_width) // 2
        base_offset_y = (canvas_height - new_height) // 2
        self._clamp_main_pan_for_size(canvas_width, canvas_height, new_width, new_height, base_offset_x, base_offset_y)

    def _clamp_main_pan_for_size(
        self,
        canvas_width: int,
        canvas_height: int,
        image_width: int,
        image_height: int,
        base_offset_x: int,
        base_offset_y: int,
    ) -> None:
        if image_width <= canvas_width:
            self.main_pan_x = 0
        else:
            min_pan_x = canvas_width - image_width - base_offset_x
            max_pan_x = -base_offset_x
            self.main_pan_x = int(np.clip(self.main_pan_x, min_pan_x, max_pan_x))

        if image_height <= canvas_height:
            self.main_pan_y = 0
        else:
            min_pan_y = canvas_height - image_height - base_offset_y
            max_pan_y = -base_offset_y
            self.main_pan_y = int(np.clip(self.main_pan_y, min_pan_y, max_pan_y))

    def _draw_empty_message(self, canvas: tk.Canvas, text: str) -> None:
        canvas.create_text(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            text=text,
            fill="#7a8493",
            font=(UI_FONT_FAMILY, 18),
        )

    def _draw_preview(self) -> None:
        tool = self.tool_var.get()
        color = "#37e66f" if self.mark_mode_var.get() in ("前景", "可能前景", "初始区域") else "#ff4d42"
        width = max(2, int(self.brush_size_var.get() * self.display_geometry.scale))

        if tool == "polygon" and self.polygon_points:
            screen_points = [self.image_to_canvas(p) for p in self.polygon_points]
            if self.drag_current is not None:
                screen_points.append(self.image_to_canvas(self.drag_current))
            flat = [coord for point in screen_points for coord in point]
            if len(screen_points) > 1:
                self.canvas.create_line(*flat, fill=color, width=2)
            for px, py in screen_points:
                self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, outline=color, fill=color)
            return

        if self.drag_start is None or self.drag_current is None or tool in ("brush", "eraser"):
            return

        start = self.image_to_canvas(self.drag_start)
        end = self.image_to_canvas(self.drag_current)
        if tool == "line":
            self.canvas.create_line(*start, *end, fill=color, width=width)
        elif tool in ("rectangle", "square", "ellipse", "circle"):
            x1, y1, x2, y2 = self._preview_bbox(tool, start, end)
            if tool in ("rectangle", "square"):
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            else:
                self.canvas.create_oval(x1, y1, x2, y2, outline=color, width=2)
        elif tool in ("pentagon", "hexagon"):
            points = self._shape_points(tool, self.drag_start, self.drag_current)
            flat = [coord for point in [self.image_to_canvas(p) for p in points] for coord in point]
            if flat:
                self.canvas.create_polygon(*flat, outline=color, fill="", width=2)

    def _preview_bbox(
        self, tool: str, start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        x1, y1 = start
        x2, y2 = end
        if tool in ("square", "circle"):
            side = max(abs(x2 - x1), abs(y2 - y1))
            x2 = x1 + side * (1 if x2 >= x1 else -1)
            y2 = y1 + side * (1 if y2 >= y1 else -1)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def main() -> int:
    try:
        require_cv2()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    root = tk.Tk()
    InteractiveSegmentationApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
