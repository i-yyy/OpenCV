from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:
    raise SystemExit("Tkinter is required to run this desktop application.") from exc


APP_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = APP_DIR / "data" / "rainbow.png"

UI_BG = "#f8f2ea"
PAPER = "#fffaf0"
PANEL = "#fff6df"
MINT = "#bdebd8"
PINK = "#f6b7c8"
ROSE = "#df6f91"
INK = "#50433b"
BROWN = "#9b735f"
DEFAULT_TEXT_LIFT = 6


@dataclass(frozen=True)
class FontChoice:
    name: str
    path: Path


FONT_CANDIDATES = (
    FontChoice("微软雅黑", Path("C:/Windows/Fonts/msyh.ttc")),
    FontChoice("微软雅黑 Bold", Path("C:/Windows/Fonts/msyhbd.ttc")),
    FontChoice("黑体", Path("C:/Windows/Fonts/simhei.ttf")),
    FontChoice("宋体", Path("C:/Windows/Fonts/simsun.ttc")),
    FontChoice("楷体", Path("C:/Windows/Fonts/simkai.ttf")),
    FontChoice("华文行楷", Path("C:/Windows/Fonts/STXINGKA.TTF")),
    FontChoice("华文彩云", Path("C:/Windows/Fonts/STCAIYUN.TTF")),
    FontChoice("华文琥珀", Path("C:/Windows/Fonts/STHUPO.TTF")),
    FontChoice("方正舒体", Path("C:/Windows/Fonts/FZSTK.TTF")),
    FontChoice("方正姚体", Path("C:/Windows/Fonts/FZYTK.TTF")),
)


TEXT_STYLES = {
    "云朵白字": {"fill": "#fffdf7", "stroke": "#70a7e8", "shadow": "#f4abc1", "stroke_width": 3},
    "粉蓝贴纸": {"fill": "#ffffff", "stroke": "#f28db0", "shadow": "#8ed6e4", "stroke_width": 4},
    "薄荷手账": {"fill": "#35685c", "stroke": "#f6f2d7", "shadow": "#9cdcc8", "stroke_width": 3},
    "暖黄艺术": {"fill": "#fff0a8", "stroke": "#9b5f8e", "shadow": "#f3a3b8", "stroke_width": 3},
    "智能协调": {"fill": None, "stroke": None, "shadow": "#f1a9bc", "stroke_width": 3},
}


def available_fonts() -> list[FontChoice]:
    fonts = [font for font in FONT_CANDIDATES if font.path.exists()]
    if not fonts:
        raise RuntimeError("No usable fonts were found in C:/Windows/Fonts.")
    return fonts


def smooth_values(values: np.ndarray, window: int = 13) -> np.ndarray:
    if values.size < 3:
        return values
    window = max(3, min(window, values.size // 2 * 2 + 1))
    pad = window // 2
    padded = np.pad(values.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def repair_occluded_outer_arc(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> list[tuple[float, float]]:
    left_x = float(xs.min())
    right_x = float(xs.max())
    left_y = float(np.max(ys[xs <= left_x + 6]))
    right_y = float(np.max(ys[xs >= right_x - 6]))
    min_y = float(ys.min())
    top_candidates = xs[ys <= min_y + 3]
    top_x = float(np.mean(top_candidates)) if top_candidates.size else float(xs[np.argmin(ys)])
    top_y = min_y

    anchors = np.asarray([[left_x, left_y], [top_x, top_y], [right_x, right_y]], dtype=float)
    matrix = np.column_stack((2 * anchors[:, 0], 2 * anchors[:, 1], np.ones(3)))
    rhs = anchors[:, 0] ** 2 + anchors[:, 1] ** 2

    try:
        center_x, center_y, constant = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        center_x = center_y = constant = float("nan")

    radius_sq = constant + center_x**2 + center_y**2
    if np.isfinite(radius_sq) and radius_sq > 0 and center_y > top_y:
        radius = math.sqrt(radius_sq)
        sample_count = max(360, int(right_x - left_x) + 1)
        sample_x = np.linspace(left_x, right_x, sample_count)
        inside = radius**2 - (sample_x - center_x) ** 2
        if np.all(inside >= 0):
            sample_y = center_y - np.sqrt(inside)
            sample_y = np.clip(sample_y, 0, height - 1)
            return [(float(x), float(y)) for x, y in zip(sample_x, sample_y)]

    sample_count = max(360, int(xs.max() - xs.min()) + 1)
    sample_x = np.linspace(float(xs.min()), float(xs.max()), sample_count)
    sample_y = np.interp(sample_x, xs, ys)

    broad_y = smooth_values(sample_y, 61)
    local_y = smooth_values(sample_y, 17)
    inward_notch = sample_y > broad_y + 18
    edge_guard = max(30, int(sample_count * 0.08))
    inward_notch[:edge_guard] = False
    inward_notch[-edge_guard:] = False

    if int(np.count_nonzero(~inward_notch)) >= 2:
        repaired_y = sample_y.copy()
        good_x = sample_x[~inward_notch]
        good_y = local_y[~inward_notch]
        repaired_y[inward_notch] = np.interp(sample_x[inward_notch], good_x, good_y)
        sample_y = repaired_y

    sample_y = smooth_values(sample_y, 21)
    sample_y = np.clip(sample_y, 0, height - 1)
    return [(float(x), float(y)) for x, y in zip(sample_x, sample_y)]


def fallback_outer_curve(width: int, height: int) -> list[tuple[float, float]]:
    left = width * 0.11
    right = width * 0.87
    center = width * 0.49
    top = height * (96 / 588)
    end_y = height * (468 / 588)
    radius = max(1.0, (right - left) / 2)
    curve = []
    for x in np.linspace(left, right, 360):
        y = top + (end_y - top) * ((x - center) / radius) ** 2
        curve.append((float(x), float(y)))
    return curve


def extract_outer_curve(image: Image.Image) -> list[tuple[float, float]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    r = rgb[:, :, 0].astype(int)
    g = rgb[:, :, 1].astype(int)
    b = rgb[:, :, 2].astype(int)

    yy, xx = np.mgrid[0:height, 0:width]
    sky_or_cloud_safe = (yy > height * 0.05) & (yy < height * 0.9) & (xx > width * 0.06) & (xx < width * 0.93)
    red_mask = (r > 185) & (g < 120) & (b < 105) & sky_or_cloud_safe
    xs: list[int] = []
    ys: list[float] = []
    for x in range(width):
        column = np.where(red_mask[:, x])[0]
        y = int(column.min()) if column.size else None
        if y is None:
            continue
        xs.append(x)
        ys.append(float(y))

    if len(xs) < 80:
        return fallback_outer_curve(width, height)

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    smoothed_ys = smooth_values(ys_arr, 17)
    points = [(float(x), float(y)) for x, y in zip(xs_arr, smoothed_ys)]

    longest: list[tuple[float, float]] = []
    current: list[tuple[float, float]] = []
    previous_x: float | None = None
    for point in points:
        if previous_x is not None and point[0] - previous_x > 3:
            if len(current) > len(longest):
                longest = current
            current = []
        current.append(point)
        previous_x = point[0]
    if len(current) > len(longest):
        longest = current

    if len(longest) < 80:
        return fallback_outer_curve(width, height)
    longest_x = np.asarray([point[0] for point in longest], dtype=float)
    longest_y = np.interp(longest_x, xs_arr, ys_arr)
    return repair_occluded_outer_arc(longest_x, longest_y, width, height)


def cumulative_lengths(points: list[tuple[float, float]]) -> list[float]:
    lengths = [0.0]
    for first, second in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(first, second))
    return lengths


def point_at_length(
    points: list[tuple[float, float]], lengths: list[float], target: float
) -> tuple[float, float, float]:
    if target <= 0:
        x1, y1 = points[0]
        x2, y2 = points[1]
        return x1, y1, math.atan2(y2 - y1, x2 - x1)
    if target >= lengths[-1]:
        x1, y1 = points[-2]
        x2, y2 = points[-1]
        return x2, y2, math.atan2(y2 - y1, x2 - x1)

    lo = 0
    hi = len(lengths) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if lengths[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    index = max(1, lo)
    prev_len = lengths[index - 1]
    next_len = lengths[index]
    ratio = 0.0 if next_len == prev_len else (target - prev_len) / (next_len - prev_len)
    x1, y1 = points[index - 1]
    x2, y2 = points[index]
    x = x1 + (x2 - x1) * ratio
    y = y1 + (y2 - y1) * ratio
    angle = math.atan2(y2 - y1, x2 - x1)
    return x, y, angle


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, stroke_width: int = 0) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)


class StickerCurveTextApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("彩虹曲线文字贴纸")
        self.root.geometry("1180x760")
        self.root.minsize(980, 660)
        self.root.configure(bg=UI_BG)

        self.fonts = available_fonts()
        self.original_image: Image.Image | None = None
        self.rendered_image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.load_time: float | None = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.active_curve: list[tuple[float, float]] = []
        self.active_lengths: list[float] = []

        self.text_var = tk.StringVar(value="东南大学")
        self.font_var = tk.StringVar(value=self.fonts[min(5, len(self.fonts) - 1)].name)
        self.size_var = tk.IntVar(value=38)
        self.spacing_var = tk.IntVar(value=6)
        self.position_var = tk.IntVar(value=50)
        self.offset_var = tk.IntVar(value=0)
        self.style_var = tk.StringVar(value="粉蓝贴纸")
        self.show_curve_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="打开图片后开始计时")
        self.time_var = tk.StringVar(value="操作时间 0.0s")

        self.build_ui()
        self.load_image(DEFAULT_IMAGE)
        self.root.after(150, self.update_timer)

    def build_ui(self) -> None:
        self.build_toolbar()

        body = tk.Frame(self.root, bg=UI_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(8, 8))

        self.panel = tk.Frame(body, bg=PANEL, width=270, highlightthickness=2, highlightbackground="#edd1a3")
        self.panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self.panel.pack_propagate(False)

        self.canvas_wrap = tk.Frame(body, bg="#e9d8c3", highlightthickness=2, highlightbackground="#dcbba0")
        self.canvas_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_wrap, bg="#fffdf7", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", lambda _event: self.refresh_preview())
        self.canvas.bind("<Button-1>", self.choose_position_from_click)

        self.build_panel()
        self.build_statusbar()

    def sticker_button(self, parent: tk.Misc, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=MINT,
            fg=INK,
            activebackground="#a9dec9",
            activeforeground=INK,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=7,
            font=("Microsoft YaHei", 10, "bold"),
            cursor="hand2",
        )

    def build_toolbar(self) -> None:
        toolbar = tk.Frame(self.root, bg=UI_BG)
        toolbar.pack(fill=tk.X, padx=14, pady=(14, 4))

        title = tk.Label(
            toolbar,
            text="Rainbow Text",
            bg=UI_BG,
            fg=INK,
            font=("Microsoft YaHei", 15, "bold"),
        )
        title.pack(side=tk.LEFT)

        time_badge = tk.Label(
            toolbar,
            textvariable=self.time_var,
            bg=PINK,
            fg="#69445a",
            font=("Microsoft YaHei", 10, "bold"),
            padx=12,
            pady=7,
        )
        time_badge.pack(side=tk.RIGHT, padx=(10, 0))

        actions = tk.Frame(toolbar, bg=UI_BG)
        actions.pack(side=tk.RIGHT)
        for label, command in (
            ("打开图片", self.open_image),
            ("保存", self.save_image),
            ("重置", self.reset_controls),
        ):
            button = self.sticker_button(actions, label, command)
            button.pack(side=tk.LEFT, padx=5)

    def label(self, parent: tk.Misc, text: str) -> tk.Label:
        return tk.Label(parent, text=text, bg=PANEL, fg=BROWN, anchor="w", font=("Microsoft YaHei", 10, "bold"))

    def build_panel(self) -> None:
        header = tk.Label(
            self.panel,
            text="贴纸参数",
            bg=PINK,
            fg="#69445a",
            font=("Microsoft YaHei", 14, "bold"),
            padx=12,
            pady=9,
        )
        header.pack(fill=tk.X, padx=14, pady=(14, 10))

        self.add_entry("文字输入（4-10字）", self.text_var)
        self.text_var.trace_add("write", lambda *_args: self.refresh_preview())
        self.add_combo("字体选择", self.font_var, [font.name for font in self.fonts])
        self.add_scale("字号", self.size_var, 20, 72)
        self.add_scale("字间距", self.spacing_var, 0, 24)
        self.add_scale("外边缘插入位置", self.position_var, 5, 95)
        self.add_scale("贴边微调", self.offset_var, -30, 30)
        self.add_combo("颜色风格", self.style_var, list(TEXT_STYLES.keys()))

        curve_check = tk.Checkbutton(
            self.panel,
            text="显示曲线预览",
            variable=self.show_curve_var,
            command=self.refresh_preview,
            bg=PANEL,
            fg=INK,
            activebackground=PANEL,
            selectcolor=PAPER,
            font=("Microsoft YaHei", 10),
        )
        curve_check.pack(anchor="w", padx=18, pady=(8, 4))

        hint = tk.Label(
            self.panel,
            text="提示：点击彩虹外边缘，可快速改变文字位置。",
            bg=PANEL,
            fg="#8a786a",
            wraplength=220,
            justify=tk.LEFT,
            font=("Microsoft YaHei", 9),
        )
        hint.pack(fill=tk.X, padx=18, pady=(10, 0))

    def add_entry(self, title: str, variable: tk.StringVar) -> None:
        self.label(self.panel, title).pack(fill=tk.X, padx=18, pady=(8, 2))
        entry = tk.Entry(
            self.panel,
            textvariable=variable,
            bg=PAPER,
            fg=INK,
            insertbackground=INK,
            relief=tk.FLAT,
            font=("Microsoft YaHei", 11),
        )
        entry.pack(fill=tk.X, padx=18, ipady=6)

    def add_combo(self, title: str, variable: tk.StringVar, values: Iterable[str]) -> None:
        self.label(self.panel, title).pack(fill=tk.X, padx=18, pady=(9, 2))
        combo = ttk.Combobox(self.panel, textvariable=variable, values=list(values), state="readonly")
        combo.pack(fill=tk.X, padx=18, ipady=3)
        combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_preview())

    def add_scale(self, title: str, variable: tk.IntVar, from_: int, to: int) -> None:
        row = tk.Frame(self.panel, bg=PANEL)
        row.pack(fill=tk.X, padx=18, pady=(9, 0))
        self.label(row, title).pack(side=tk.LEFT)
        value_label = tk.Label(row, textvariable=variable, bg=PANEL, fg=INK, font=("Microsoft YaHei", 9))
        value_label.pack(side=tk.RIGHT)
        scale = tk.Scale(
            self.panel,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=lambda _value: self.refresh_preview(),
            bg=PANEL,
            fg=INK,
            troughcolor="#f0dfc7",
            activebackground=PINK,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        scale.pack(fill=tk.X, padx=16)

    def build_statusbar(self) -> None:
        statusbar = tk.Frame(self.root, bg="#efe0ce", height=34)
        statusbar.pack(fill=tk.X, padx=14, pady=(0, 12))
        statusbar.pack_propagate(False)
        tk.Label(statusbar, textvariable=self.status_var, bg="#efe0ce", fg="#7d6c62", font=("Microsoft YaHei", 10)).pack(
            side=tk.LEFT, padx=12
        )

    def selected_font(self) -> FontChoice:
        for font in self.fonts:
            if font.name == self.font_var.get():
                return font
        return self.fonts[0]

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择背景图片",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")],
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            messagebox.showerror("打开失败", f"无法读取图片：\n{path}\n\n{exc}")
            return
        self.original_image = image
        self.load_time = time.perf_counter()
        self.status_var.set(f"已读取图片：{path.name}")
        self.refresh_preview()

    def validate_text(self) -> str | None:
        text = self.text_var.get().strip()
        if not 4 <= len(text) <= 10:
            self.status_var.set("文字需要控制在 4-10 个字之间")
            return None
        return text

    def render_result(self) -> Image.Image | None:
        if self.original_image is None:
            return None
        text = self.validate_text()
        if text is None:
            return self.original_image.copy()

        image = self.original_image.convert("RGBA")
        curve = extract_outer_curve(self.original_image)
        lengths = cumulative_lengths(curve)
        self.active_curve = curve
        self.active_lengths = lengths

        font = ImageFont.truetype(str(self.selected_font().path), self.size_var.get())
        style = TEXT_STYLES[self.style_var.get()]
        fill = style["fill"]
        stroke = style["stroke"]
        if fill is None or stroke is None:
            fill, stroke = self.smart_colors(image, curve, lengths)
        shadow = style["shadow"]
        stroke_width = int(style["stroke_width"])

        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        char_widths = []
        for char in text:
            bbox = text_bbox(probe, char, font, stroke_width)
            char_widths.append(max(1, bbox[2] - bbox[0]))
        spacing = self.spacing_var.get()
        total_width = sum(char_widths) + spacing * (len(text) - 1)
        center_length = lengths[-1] * self.position_var.get() / 100
        start = center_length - total_width / 2

        cursor = start
        for index, char in enumerate(text):
            char_center = cursor + char_widths[index] / 2
            if 0 <= char_center <= lengths[-1]:
                x, y, angle = point_at_length(curve, lengths, char_center)
                normal_angle = angle - math.pi / 2
                visual_offset = self.offset_var.get() + DEFAULT_TEXT_LIFT
                x += math.cos(normal_angle) * visual_offset
                y += math.sin(normal_angle) * visual_offset
                rotate_angle = -math.degrees(angle)
                self.paste_rotated_char(image, char, font, x, y, rotate_angle, fill, stroke, shadow, stroke_width)
            cursor += char_widths[index] + spacing

        self.status_var.set(
            f"当前字体：{self.font_var.get()} | 曲线：彩虹外边缘 | 风格：{self.style_var.get()}"
        )
        return image.convert("RGB")

    def smart_colors(
        self, image: Image.Image, curve: list[tuple[float, float]], lengths: list[float]
    ) -> tuple[str, str]:
        center_length = lengths[-1] * self.position_var.get() / 100
        x, y, _angle = point_at_length(curve, lengths, center_length)
        arr = np.asarray(image.convert("RGB"))
        height, width = arr.shape[:2]
        x1 = max(0, int(x) - 18)
        x2 = min(width, int(x) + 18)
        y1 = max(0, int(y) - 18)
        y2 = min(height, int(y) + 18)
        patch = arr[y1:y2, x1:x2]
        if patch.size == 0:
            return "#ffffff", "#70a7e8"
        brightness = float(np.mean(patch))
        if brightness > 185:
            return "#62495c", "#ffffff"
        return "#fffdf7", "#315f9f"

    def paste_rotated_char(
        self,
        image: Image.Image,
        char: str,
        font: ImageFont.ImageFont,
        x: float,
        y: float,
        angle: float,
        fill: str,
        stroke: str,
        shadow: str,
        stroke_width: int,
    ) -> None:
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bbox = text_bbox(probe, char, font, stroke_width)
        width = bbox[2] - bbox[0] + stroke_width * 4 + 16
        height = bbox[3] - bbox[1] + stroke_width * 4 + 16
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        tile = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(tile)
        tx = 8 + stroke_width * 2 - bbox[0]
        ty = 8 + stroke_width * 2 - bbox[1]
        anchor_x = 8 + stroke_width * 2 + text_width / 2
        anchor_y = 8 + stroke_width * 2 + text_height
        draw.text((tx + 2, ty + 3), char, font=font, fill=shadow, stroke_width=stroke_width, stroke_fill=shadow)
        draw.text((tx, ty), char, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke)
        rotated = tile.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        anchor_mask = Image.new("L", (width, height), 0)
        anchor_draw = ImageDraw.Draw(anchor_mask)
        anchor_draw.ellipse((anchor_x - 1, anchor_y - 1, anchor_x + 1, anchor_y + 1), fill=255)
        rotated_anchor = anchor_mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
        anchor_pixels = np.argwhere(np.asarray(rotated_anchor) > 0)
        if anchor_pixels.size:
            rotated_anchor_y, rotated_anchor_x = anchor_pixels.mean(axis=0)
        else:
            rotated_anchor_x = rotated.width / 2
            rotated_anchor_y = rotated.height / 2
        image.alpha_composite(rotated, (int(x - rotated_anchor_x), int(y - rotated_anchor_y)))

    def refresh_preview(self) -> None:
        if self.original_image is None or self.canvas.winfo_width() <= 1:
            return
        self.rendered_image = self.render_result()
        if self.rendered_image is None:
            return
        preview = self.rendered_image.convert("RGBA")
        if self.show_curve_var.get() and self.active_curve:
            self.draw_curve_preview(preview)
        self.show_on_canvas(preview.convert("RGB"))

    def draw_curve_preview(self, image: Image.Image) -> None:
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        if len(self.active_curve) > 1:
            draw.line(self.active_curve, fill=(241, 125, 157, 210), width=4)
            for i, point in enumerate(self.active_curve[::34]):
                x, y = point
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(189, 235, 216, 230))
            if self.active_lengths:
                x, y, _angle = point_at_length(
                    self.active_curve, self.active_lengths, self.active_lengths[-1] * self.position_var.get() / 100
                )
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 250, 240, 255), outline=(223, 111, 145, 255), width=3)
        image.alpha_composite(overlay)

    def show_on_canvas(self, image: Image.Image) -> None:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        scale = min(canvas_width / image.width, canvas_height / image.height, 1.0)
        display_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        display = image.resize(display_size, Image.Resampling.LANCZOS)
        self.display_scale = scale
        self.display_offset = ((canvas_width - display_size[0]) // 2, (canvas_height - display_size[1]) // 2)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#fffdf7", outline="")
        self.draw_canvas_dots(canvas_width, canvas_height)
        self.canvas.create_image(self.display_offset[0], self.display_offset[1], image=self.photo, anchor=tk.NW)

    def draw_canvas_dots(self, width: int, height: int) -> None:
        for x in range(18, width, 36):
            for y in range(18, height, 36):
                self.canvas.create_oval(x, y, x + 2, y + 2, fill="#f1dfca", outline="")

    def choose_position_from_click(self, event: tk.Event) -> None:
        if not self.active_curve or not self.active_lengths:
            return
        ox, oy = self.display_offset
        if self.display_scale <= 0:
            return
        x = (event.x - ox) / self.display_scale
        y = (event.y - oy) / self.display_scale
        best_index = min(range(len(self.active_curve)), key=lambda i: (self.active_curve[i][0] - x) ** 2 + (self.active_curve[i][1] - y) ** 2)
        percent = int(round(self.active_lengths[best_index] / self.active_lengths[-1] * 100))
        self.position_var.set(max(5, min(95, percent)))
        self.refresh_preview()

    def reset_controls(self) -> None:
        self.text_var.set("彩虹快乐日")
        self.font_var.set(self.fonts[min(5, len(self.fonts) - 1)].name)
        self.size_var.set(38)
        self.spacing_var.set(6)
        self.position_var.set(50)
        self.offset_var.set(0)
        self.style_var.set("粉蓝贴纸")
        self.show_curve_var.set(False)
        self.refresh_preview()

    def save_image(self) -> None:
        if self.rendered_image is None:
            messagebox.showwarning("没有结果", "请先打开图片并生成预览。")
            return
        path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return
        self.rendered_image.save(path)
        self.status_var.set(f"已保存：{Path(path).name}")

    def update_timer(self) -> None:
        if self.load_time is None:
            self.time_var.set("操作时间 0.0s")
        else:
            elapsed = time.perf_counter() - self.load_time
            self.time_var.set(f"操作时间 {elapsed:.1f}s")
        self.root.after(200, self.update_timer)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TCombobox", fieldbackground=PAPER, background=PAPER, foreground=INK, arrowcolor=INK)
    StickerCurveTextApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
