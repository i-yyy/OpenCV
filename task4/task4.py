"""Task 4: interactive CPU-only front-view correction with LivePortrait ONNX."""
from __future__ import annotations

import json
import gc
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from liveportrait_backend import LivePortraitCPU


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DEFAULT_FRONTAL_REFERENCE = DATA / "face-正.png"


class PortraitApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("侧脸转正脸 · CPU 深度模型交互实验")
        self.geometry("1450x820")
        self.minsize(1120, 680)
        self.backend: LivePortraitCPU | None = None
        self.backend_kind = ""
        self.refiner = None
        self.flux2_gpu = None
        self.flux2_cpu = None
        self.source_path: Path | None = None
        self.original: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.crop_result: np.ndarray | None = None
        self.source_box: tuple[int, int, int, int] | None = None
        self.source_yaw = 0.0
        self.source_pitch = 0.0
        self.source_roll = 0.0
        self.cumulative = 0.0
        self.history: list[float] = []
        self.session_started: float | None = None
        self.stopped_elapsed: float | None = None
        self.last_infer = 0.0
        self.was_refined = False
        self.localized_refined_side: str | None = None
        self.frontal_reference_used = False
        self.reference_guided_used = False
        self.busy = False
        self.messages: queue.Queue = queue.Queue()
        self._build_ui()
        self.after(80, self._poll)
        self.after(200, self._tick)
        default = DATA / "face.png"
        if default.exists():
            self.after(250, lambda: self.load_image(default, "human"))

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 12, "bold"))
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=4)
        main.columnconfigure(2, weight=2)
        main.rowconfigure(1, weight=1)
        ttk.Label(main, text="原图（拖动鼠标框选头部）", style="Title.TLabel").grid(row=0, column=0)
        ttk.Label(main, text="当前生成结果", style="Title.TLabel").grid(row=0, column=1)
        ttk.Label(main, text="交互操作", style="Title.TLabel").grid(row=0, column=2)
        self.left = tk.Canvas(main, bg="#222", highlightthickness=1, highlightbackground="#777")
        self.right = tk.Canvas(main, bg="#222", highlightthickness=1, highlightbackground="#777")
        self.left.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=8)
        self.right.grid(row=1, column=1, sticky="nsew", padx=(0, 8), pady=8)
        self.left.bind("<ButtonPress-1>", self._drag_start)
        self.left.bind("<B1-Motion>", self._drag_move)
        self.left.bind("<ButtonRelease-1>", self._drag_end)
        self.left.bind("<Configure>", lambda _e: self._draw_images())
        self.right.bind("<Configure>", lambda _e: self._draw_images())

        # The controls exceed a typical laptop-height window, so keep them in
        # a real scrollable viewport instead of clipping the lower buttons.
        controls = ttk.Frame(main)
        controls.grid(row=1, column=2, sticky="nsew", pady=8)
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(0, weight=1)
        self.control_canvas = tk.Canvas(controls, highlightthickness=0, borderwidth=0)
        control_scrollbar = ttk.Scrollbar(controls, orient="vertical", command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=control_scrollbar.set)
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        control_scrollbar.grid(row=0, column=1, sticky="ns")
        panel = ttk.Frame(self.control_canvas, padding=(8, 4))
        self._control_window = self.control_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda _e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        self.control_canvas.bind("<Configure>", lambda e: self.control_canvas.itemconfigure(self._control_window, width=e.width))
        self.bind_all("<MouseWheel>", self._scroll_controls, add="+")
        panel.columnconfigure(1, weight=1)
        ttk.Button(panel, text="打开动漫人物图", command=lambda: self._choose("human")).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(panel, text="打开小猫图", command=lambda: self._choose("animal")).grid(row=0, column=1, sticky="ew", pady=2)
        self.kind_var = tk.StringVar(value="人物/动漫")
        ttk.Label(panel, text="当前模型：").grid(row=1, column=0, sticky="w", pady=(12, 2))
        ttk.Label(panel, textvariable=self.kind_var).grid(row=1, column=1, sticky="w", pady=(12, 2))
        ttk.Label(panel, text="模型估计 yaw / pitch / roll：").grid(row=2, column=0, sticky="w")
        self.estimated_var = tk.StringVar(value="--")
        ttk.Label(panel, textvariable=self.estimated_var).grid(row=2, column=1, sticky="w")
        ttk.Label(panel, text="校准原始 yaw (°)：").grid(row=3, column=0, sticky="w", pady=(8, 2))
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(panel, textvariable=self.source_var, width=9).grid(row=3, column=1, sticky="ew", pady=(8, 2))
        ttk.Label(panel, text="本次相对扭转角 (°)：").grid(row=4, column=0, sticky="w", pady=(8, 2))
        self.delta_var = tk.StringVar(value="10")
        ttk.Entry(panel, textvariable=self.delta_var, width=9).grid(row=4, column=1, sticky="ew", pady=(8, 2))
        ttk.Label(panel, text="pitch 校正比例 (0~1)：").grid(row=5, column=0, sticky="w", pady=(8, 2))
        self.pitch_correction_var = tk.StringVar(value="0.55")
        ttk.Entry(panel, textvariable=self.pitch_correction_var, width=9).grid(row=5, column=1, sticky="ew", pady=(8, 2))
        ttk.Label(panel, text="roll 校正比例 (0~1)：").grid(row=6, column=0, sticky="w", pady=(8, 2))
        self.roll_correction_var = tk.StringVar(value="0.70")
        ttk.Entry(panel, textvariable=self.roll_correction_var, width=9).grid(row=6, column=1, sticky="ew", pady=(8, 2))
        ttk.Button(panel, text="向左扭转", command=lambda: self.apply_delta(-1)).grid(row=7, column=0, sticky="ew", pady=3)
        ttk.Button(panel, text="向右扭转", command=lambda: self.apply_delta(1)).grid(row=7, column=1, sticky="ew", pady=3)
        ttk.Button(panel, text="一步转到 0° 正脸", command=self.to_front).grid(row=8, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(panel, text="撤销上一步", command=self.undo).grid(row=9, column=0, sticky="ew", pady=3)
        ttk.Button(panel, text="重置角度", command=self.reset_angle).grid(row=9, column=1, sticky="ew", pady=3)
        ttk.Button(panel, text="按当前框重新准备", command=self.reprepare).grid(row=10, column=0, columnspan=2, sticky="ew", pady=(10, 3))
        self.angle_var = tk.StringVar(value="累计：0.0°   目标：--")
        ttk.Label(panel, textvariable=self.angle_var, font=("Microsoft YaHei UI", 10, "bold")).grid(row=11, column=0, columnspan=2, sticky="w", pady=(12, 3))
        self.time_var = tk.StringVar(value="计时：00:00.0")
        ttk.Label(panel, textvariable=self.time_var).grid(row=12, column=0, columnspan=2, sticky="w")
        self.perf_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.perf_var, wraplength=280).grid(row=13, column=0, columnspan=2, sticky="w", pady=4)
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.grid(row=14, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Separator(panel).grid(row=15, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        ttk.Label(panel, text="局部修复强度：").grid(row=16, column=0, sticky="w")
        self.strength_var = tk.StringVar(value="0.62")
        ttk.Entry(panel, textvariable=self.strength_var, width=9).grid(row=16, column=1, sticky="ew")
        ttk.Label(panel, text="随机种子：").grid(row=17, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="2026")
        ttk.Entry(panel, textvariable=self.seed_var, width=9).grid(row=17, column=1, sticky="ew")
        ttk.Button(panel, text="局部五官修复（CPU 较慢）", command=self.refine).grid(row=18, column=0, columnspan=2, sticky="ew", pady=(5, 3))
        self.use_reference_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="修复时参考同角色正脸图（仅动漫）", variable=self.use_reference_var).grid(row=19, column=0, columnspan=2, sticky="w", pady=(5, 2))
        ttk.Label(panel, text="正脸参考引导强度：").grid(row=20, column=0, sticky="w", pady=(2, 2))
        self.reference_scale_var = tk.StringVar(value="0.55")
        ttk.Entry(panel, textvariable=self.reference_scale_var, width=9).grid(row=20, column=1, sticky="ew", pady=(2, 2))
        ttk.Button(panel, text="生图模型生成正脸候选（自动区分动漫/小猫）", command=self.generate_candidate).grid(row=21, column=0, columnspan=2, sticky="ew", pady=(3, 4))
        ttk.Button(panel, text="FLUX.2 Klein 原图编辑生成正脸1" , command=self.generate_flux2_candidate).grid(row=31, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        ttk.Button(panel, text="FLUX.2 Klein 原图编辑生成正脸2", command=lambda: self.generate_flux2_candidate(use_cpu=True)).grid(row=32, column=0, columnspan=2, sticky="ew", pady=(3, 4))
        ttk.Label(panel, text="动漫局部修复侧：").grid(row=26, column=0, sticky="w", pady=(8, 2))
        self.anime_side_var = tk.StringVar(value="left")
        ttk.Combobox(panel, textvariable=self.anime_side_var, values=("left", "right"), state="readonly", width=9).grid(row=26, column=1, sticky="ew", pady=(8, 2))
        ttk.Button(panel, text="只修复选定半脸（CPU 较慢）", command=self.refine_anime_half).grid(row=27, column=0, columnspan=2, sticky="ew", pady=(3, 8))
        ttk.Button(panel, text="满意：停止计时并保存", command=self.accept).grid(row=28, column=0, columnspan=2, sticky="ew", pady=(8, 3))
        ttk.Button(panel, text="仅保存候选图", command=lambda: self.save(False)).grid(row=29, column=0, columnspan=2, sticky="ew", pady=3)
        self.status_var = tk.StringVar(value="请打开图像。首次加载模型需要一些时间。")
        ttk.Label(panel, textvariable=self.status_var, wraplength=280, foreground="#174c84").grid(row=30, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _scroll_controls(self, event) -> None:
        """Scroll the right-side control viewport with the mouse wheel."""
        if hasattr(self, "control_canvas"):
            self.control_canvas.yview_scroll(-max(1, int(event.delta / 120)), "units")

    def _choose(self, kind: str) -> None:
        path = filedialog.askopenfilename(initialdir=DATA, filetypes=[("图像", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.load_image(Path(path), kind)

    def load_image(self, path: Path, kind: str) -> None:
        self.session_started, self.stopped_elapsed = time.perf_counter(), None
        if self.backend_kind and self.backend_kind != kind:
            self.backend = None
            self.refiner = None
            gc.collect()
        raw = np.fromfile(str(path), np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("读取失败", str(path)); return
        self.source_path, self.original = path, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.result = self.original.copy()
        h, w = self.original.shape[:2]
        # Tuned defaults only initialize the editable crop; users can drag a new square at any time.
        if path.name.lower() == "face.png":
            self.source_box = (0, 0, min(w, 480), min(h, 480))
        elif path.name.lower() == "cat.jpg":
            self.source_box = (220, 0, min(w, 900), min(h, 680))
        else:
            side = min(h, w); self.source_box = ((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2)
        self.kind_var.set("动漫人物（Human）" if kind == "human" else "小猫（Animal）")
        self.strength_var.set("0.55" if kind == "human" else "0.62")
        self.seed_var.set("2028" if kind == "human" else "2026")
        self.pending_kind = kind
        self.cumulative, self.history = 0.0, []
        self.was_refined = False
        self.localized_refined_side = None
        self.frontal_reference_used = False
        self.reference_guided_used = False
        self.use_reference_var.set(kind == "human")
        self._draw_images()
        self._prepare_async()

    def _prepare_async(self) -> None:
        if self.busy or self.source_path is None: return
        kind, path, box = self.pending_kind, self.source_path, self.source_box
        self._set_busy(True, "正在加载 CPU 模型并提取原图特征…")
        def work():
            try:
                if self.backend is None or self.backend_kind != kind:
                    self.backend = None
                    gc.collect()
                    self.backend = LivePortraitCPU(MODELS, kind, max(1, (os.cpu_count() or 4) // 2))
                    self.backend_kind = kind
                state = self.backend.prepare(path, box)
                self.messages.put(("prepared", state))
            except Exception as exc: self.messages.put(("error", exc))
        threading.Thread(target=work, daemon=True).start()

    def reprepare(self) -> None:
        self.cumulative, self.history = 0.0, []
        self._prepare_async()

    def apply_delta(self, sign: int) -> None:
        if self.busy: return
        try: delta = abs(float(self.delta_var.get())) * sign
        except ValueError: messagebox.showwarning("角度无效", "请输入数字角度。"); return
        self.history.append(self.cumulative)
        self.cumulative = float(np.clip(self.cumulative + delta, -90, 90))
        self._render_async()

    def to_front(self) -> None:
        if self.busy: return
        try: base = float(self.source_var.get())
        except ValueError: messagebox.showwarning("角度无效", "校准原始 yaw 必须是数字。"); return
        self.history.append(self.cumulative)
        self.cumulative = float(np.clip(-base, -90, 90))
        self._render_async()

    def undo(self) -> None:
        if self.busy: return
        if not self.history: return
        self.cumulative = self.history.pop(); self._render_async()

    def reset_angle(self) -> None:
        if self.busy: return
        self.history.append(self.cumulative); self.cumulative = 0.0; self._render_async()

    def _render_async(self) -> None:
        if self.busy or self.backend is None or self.backend.state is None: return
        try: base = float(self.source_var.get())
        except ValueError: messagebox.showwarning("角度无效", "校准原始 yaw 必须是数字。"); return
        try:
            pitch_correction = float(self.pitch_correction_var.get())
            roll_correction = float(self.roll_correction_var.get())
        except ValueError:
            messagebox.showwarning("校正比例无效", "pitch / roll 校正比例必须是数字。"); return
        if not (0.0 <= pitch_correction <= 1.0 and 0.0 <= roll_correction <= 1.0):
            messagebox.showwarning("校正比例无效", "pitch / roll 校正比例必须位于 0 到 1 之间。"); return
        target = float(np.clip(base + self.cumulative, -90, 90))
        self.angle_var.set(f"累计：{self.cumulative:+.1f}°   目标：{target:+.1f}°")
        self._set_busy(True, f"CPU 正在生成目标 yaw {target:+.1f}°…")
        def work():
            try: self.messages.put(("rendered", target, *self.backend.render(target, pitch_correction, roll_correction)))
            except Exception as exc: self.messages.put(("error", exc))
        threading.Thread(target=work, daemon=True).start()

    def refine(self) -> None:
        if self.busy or self.crop_result is None or self.backend is None:
            if self.crop_result is None:
                messagebox.showinfo("请先扭转", "请先生成一个角度预览，再进行高质量修复。")
            return
        try:
            strength = float(self.strength_var.get()); seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showwarning("参数无效", "修复强度和随机种子必须是数字。"); return
        use_reference = self.backend_kind == "human" and self.use_reference_var.get()
        try:
            reference_scale = float(self.reference_scale_var.get())
            if not 0.0 <= reference_scale <= 1.0: raise ValueError
        except ValueError:
            messagebox.showwarning("参考强度无效", "正脸参考引导强度必须在 0 到 1 之间。"); return
        self._set_busy(True, "正在 CPU 参考引导修复五官；首次加载可能需要数分钟…" if use_reference else "正在 CPU 扩散修复头部；首次加载可能需要数分钟…")
        crop = self.crop_result.copy(); kind = self.backend_kind
        def work():
            try:
                if self.refiner is None:
                    from refiner_backend import DomainRefinerCPU
                    self.refiner = DomainRefinerCPU()
                reference = DEFAULT_FRONTAL_REFERENCE if use_reference else None
                refined, elapsed = self.refiner.refine(crop, kind, seed=seed, strength=strength,
                                                       reference_path=reference, reference_scale=reference_scale)
                composed = self.backend.compose(refined)
                self.messages.put(("refined", refined, composed, elapsed, self.refiner.load_seconds, use_reference))
            except Exception as exc: self.messages.put(("error", exc))
        threading.Thread(target=work, daemon=True).start()

    def generate_candidate(self) -> None:
        if self.busy or self.crop_result is None or self.backend is None:
            messagebox.showinfo("请先扭转", "请先生成正脸预览，再用生图模型生成候选。")
            return
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showwarning("参数无效", "随机种子必须是整数。")
            return
        crop, kind = self.crop_result.copy(), self.backend_kind
        self._set_busy(True, "正在用生图模型生成独立正脸候选；CPU 上可能需要数分钟…")
        def work():
            try:
                if self.refiner is None:
                    from refiner_backend import DomainRefinerCPU
                    self.refiner = DomainRefinerCPU()
                generated, elapsed = self.refiner.generate_frontal_candidate(crop, kind, seed)
                self.messages.put(("direct_generated", generated, self.backend.compose(generated), elapsed, self.refiner.load_seconds, kind))
            except Exception as exc:
                self.messages.put(("error", exc))
        threading.Thread(target=work, daemon=True).start()

    def _compose_original_crop(self, crop: np.ndarray) -> np.ndarray:
        """Feather a crop into the original image without requiring LivePortrait state."""
        if self.original is None or self.source_box is None:
            raise RuntimeError("尚未加载原图或选择头部区域")
        x0, y0, x1, y1 = self.source_box
        resized = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LANCZOS4)
        result = self.original.copy()
        h, w = resized.shape[:2]
        edge = max(4, min(h, w) // 24)
        alpha = np.ones((h, w), np.float32)
        alpha[:edge] *= np.linspace(0, 1, edge)[:, None]
        alpha[-edge:] *= np.linspace(1, 0, edge)[:, None]
        alpha[:, :edge] *= np.linspace(0, 1, edge)[None, :]
        alpha[:, -edge:] *= np.linspace(1, 0, edge)[None, :]
        roi = result[y0:y1, x0:x1]
        result[y0:y1, x0:x1] = (resized * alpha[..., None] + roi * (1 - alpha[..., None])).astype(np.uint8)
        return result

    def generate_flux2_candidate(self, use_cpu: bool = False) -> None:
        """Run FLUX.2 Klein's native full-image reference-edit workflow."""
        if self.busy or self.original is None:
            messagebox.showinfo("FLUX.2", "Please open an anime or cat source image first.")
            return
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showwarning("FLUX.2", "The random seed must be an integer.")
            return
        source = self.original.copy()
        device = "CPU" if use_cpu else ""
        kind = getattr(self, "pending_kind", self.backend_kind or "human")
        self._set_busy(
            True,
            "FLUX.2 Klein is editing from the full original image locally; it does not use LivePortrait first."
            if not use_cpu else
            "FLUX.2 Klein is editing from the full original image locally on CPU; it does not use LivePortrait first.",
        )

        def work() -> None:
            try:
                if use_cpu and self.flux2_cpu is None:
                    from flux2_backend import Flux2KleinGPU
                    self.flux2_cpu = Flux2KleinGPU("http://127.0.0.1:8192")
                elif not use_cpu and self.flux2_gpu is None:
                    from flux2_backend import Flux2KleinGPU
                    self.flux2_gpu = Flux2KleinGPU("http://127.0.0.1:8191")
                client = self.flux2_cpu if use_cpu else self.flux2_gpu
                generated, elapsed = client.generate(source, kind, seed)
                generated = cv2.resize(generated, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                if self.source_box is not None:
                    x0, y0, x1, y1 = self.source_box
                    crop = generated[y0:y1, x0:x1].copy()
                else:
                    crop = generated.copy()
                self.messages.put(("flux2_generated", crop, generated, elapsed, kind, device))
            except Exception as exc:
                self.messages.put(("error", exc))

        threading.Thread(target=work, daemon=True).start()

    def refine_anime_half(self) -> None:
        """Restyle one bad half; retain the other LivePortrait half unchanged."""
        if self.backend_kind != "human":
            messagebox.showinfo("仅限动漫人物", "“只修复选定半脸”仅适用于动漫人物；小猫请使用“局部五官修复”。")
            return
        if self.busy or self.crop_result is None or self.backend is None:
            if self.crop_result is None:
                messagebox.showinfo("请先扭转", "请先生成一个正脸预览，再进行半脸动漫修复。")
            return
        try:
            strength = float(self.strength_var.get())
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showwarning("参数无效", "修复强度和随机种子必须是数字。")
            return
        side = self.anime_side_var.get()
        use_reference = self.use_reference_var.get()
        try:
            reference_scale = float(self.reference_scale_var.get())
            if not 0.0 <= reference_scale <= 1.0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("参考强度无效", "正脸参考引导强度必须在 0 到 1 之间。")
            return
        self._set_busy(
            True,
            f"正在以正脸参考修复 {side} 半脸；另一侧保持当前结果…" if use_reference
            else f"正在仅修复 {side} 半脸为动漫风格；另一侧保持当前结果…",
        )
        crop = self.crop_result.copy()
        def work():
            try:
                if self.refiner is None:
                    from refiner_backend import DomainRefinerCPU
                    self.refiner = DomainRefinerCPU()
                reference = DEFAULT_FRONTAL_REFERENCE if use_reference else None
                refined, elapsed = self.refiner.refine_anime_half(
                    crop, side=side, seed=seed, strength=strength,
                    reference_path=reference, reference_scale=reference_scale,
                )
                composed = self.backend.compose(refined)
                self.messages.put(("anime_half_refined", side, refined, composed, elapsed,
                                   self.refiner.load_seconds, use_reference))
            except Exception as exc:
                self.messages.put(("error", exc))
        threading.Thread(target=work, daemon=True).start()

    def show_frontal_reference_candidate(self) -> None:
        """Show the paired front view as an independent, never-blended candidate."""
        if self.backend_kind != "human":
            messagebox.showinfo("仅限动漫人物", "同角色正脸参考候选仅适用于动漫人物，不会用于小猫。")
            return
        if self.busy:
            return
        if not DEFAULT_FRONTAL_REFERENCE.exists():
            messagebox.showerror("缺少参考图", f"未找到同角色正脸参考图：\n{DEFAULT_FRONTAL_REFERENCE}")
            return
        self._set_busy(True, "正在读取同角色正脸参考候选…")
        def work():
            try:
                raw = np.fromfile(str(DEFAULT_FRONTAL_REFERENCE), dtype=np.uint8)
                bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if bgr is None: raise ValueError(f"无法读取正脸参考图：{DEFAULT_FRONTAL_REFERENCE}")
                self.messages.put(("frontal_reference", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            except Exception as exc:
                self.messages.put(("error", exc))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                msg = self.messages.get_nowait()
                if msg[0] == "prepared":
                    s = msg[1]; self.source_yaw, self.source_pitch, self.source_roll = s.yaw, s.pitch, s.roll; self.source_var.set(f"{s.yaw:.1f}")
                    self.estimated_var.set(f"{s.yaw:+.1f}° / {s.pitch:+.1f}° / {s.roll:+.1f}°")
                    self.angle_var.set(f"累计：0.0°   目标：{s.yaw:+.1f}°")
                    self.perf_var.set(f"模型加载 {self.backend.load_seconds:.1f}s；读取 {s.read_seconds:.3f}s；特征准备 {s.prepare_seconds:.1f}s")
                    self._set_busy(False, "原图特征已缓存。输入角度后可反复扭转。")
                elif msg[0] == "rendered":
                    _, target, crop, result, elapsed, target_pitch, target_roll = msg
                    self.crop_result, self.result, self.last_infer = crop, result, elapsed
                    self.was_refined = False
                    self.localized_refined_side = None
                    self.frontal_reference_used = False
                    self.reference_guided_used = False
                    self.perf_var.set(self.perf_var.get().split("；本轮")[0] + f"；本轮 {elapsed:.1f}s")
                    self._set_busy(False, f"已生成 yaw {target:+.1f}° / pitch {target_pitch:+.1f}° / roll {target_roll:+.1f}°。请观察五官，继续微调或点击满意。")
                    self._draw_images()
                elif msg[0] == "refined":
                    _, crop, result, elapsed, load_elapsed, used_reference = msg
                    self.crop_result, self.result, self.last_infer = crop, result, elapsed
                    self.was_refined = True
                    self.frontal_reference_used = False
                    self.reference_guided_used = used_reference
                    self.perf_var.set(f"扩散模型加载 {load_elapsed:.1f}s；本轮高质量修复 {elapsed:.1f}s")
                    self._set_busy(False, "正脸参考引导修复完成。可降低参考强度或更换种子生成候选。" if used_reference else "领域修复完成。可换随机种子再次生成，或点击满意。")
                    self._draw_images()
                elif msg[0] == "anime_half_refined":
                    _, side, crop, result, elapsed, load_elapsed, used_reference = msg
                    self.crop_result, self.result, self.last_infer = crop, result, elapsed
                    self.was_refined = True
                    self.localized_refined_side = side
                    self.frontal_reference_used = False
                    self.reference_guided_used = used_reference
                    self.perf_var.set(f"动漫模型加载 {load_elapsed:.1f}s；本轮 {side} 半脸修复 {elapsed:.1f}s")
                    self._set_busy(
                        False,
                        f"{side} 半脸正脸参考引导修复完成；另一半保持当前旋转结果。" if used_reference
                        else f"{side} 半脸动漫修复完成；另一半保持当前旋转结果。",
                    )
                    self._draw_images()
                elif msg[0] == "direct_generated":
                    _, crop, result, elapsed, load_elapsed, kind = msg
                    self.crop_result, self.result, self.last_infer = crop, result, elapsed
                    self.was_refined = True
                    self.localized_refined_side = None
                    self.frontal_reference_used = self.reference_guided_used = False
                    name = "动漫生图模型" if kind == "human" else "小猫生图模型"
                    self.perf_var.set(f"{name}加载 {load_elapsed:.1f}s；候选生成 {elapsed:.1f}s")
                    self._set_busy(False, f"{name}候选已生成；原 LivePortrait 扭转流程未改变。")
                    self._draw_images()
                elif msg[0] == "flux2_generated":
                    _, crop, result, elapsed, kind, device = msg
                    self.crop_result, self.result, self.last_infer = crop, result, elapsed
                    self.was_refined = True
                    self.localized_refined_side = None
                    self.frontal_reference_used = self.reference_guided_used = False
                    label = "anime" if kind == "human" else "cat"
                    device_label = "CPU " if device else ""
                    self.perf_var.set(f"FLUX.2 Klein {device_label}{label} reference-edit frontal candidate {elapsed:.1f}s (4 steps)")
                    self._set_busy(False, f"FLUX.2 {device_label}completed native full-image reference editing; existing LivePortrait paths were not changed.")
                    self._draw_images()
                elif msg[0] == "frontal_reference":
                    _, result = msg
                    self.crop_result, self.result, self.last_infer = None, result, 0.0
                    self.was_refined, self.frontal_reference_used = False, True
                    self.reference_guided_used = False
                    self.localized_refined_side = None
                    self.perf_var.set("同角色正脸参考候选（未与生成结果融合）")
                    self._set_busy(False, "正在显示同角色正脸参考候选。点击角度按钮可随时回到 LivePortrait 结果。")
                    self._draw_images()
                elif msg[0] == "error":
                    self._set_busy(False, "运行失败")
                    messagebox.showerror("运行失败", str(msg[1]))
        except queue.Empty: pass
        self.after(80, self._poll)

    def _set_busy(self, value: bool, text: str) -> None:
        self.busy = value; self.status_var.set(text)
        if value: self.progress.start(12)
        else: self.progress.stop()

    def _tick(self) -> None:
        if self.session_started is not None:
            elapsed = self.stopped_elapsed if self.stopped_elapsed is not None else time.perf_counter() - self.session_started
            self.time_var.set(f"计时：{int(elapsed//60):02d}:{elapsed%60:04.1f}")
        self.after(100, self._tick)

    def accept(self) -> None:
        if self.session_started is not None and self.stopped_elapsed is None:
            self.stopped_elapsed = time.perf_counter() - self.session_started
        self.save(True)

    def save(self, accepted: bool) -> None:
        if self.result is None or self.source_path is None: return
        RESULTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{self.source_path.stem}_{self.backend_kind}_{stamp}"
        image_path = RESULTS / f"{stem}.png"
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR))
        if ok: encoded.tofile(str(image_path))
        elapsed = self.stopped_elapsed or (time.perf_counter() - self.session_started if self.session_started else 0)
        metadata = {"source": str(self.source_path), "model": self.backend_kind, "provider": "CPUExecutionProvider",
                    "estimated_source_yaw": self.source_yaw, "calibrated_source_yaw": float(self.source_var.get()),
                    "estimated_source_pitch": self.source_pitch, "estimated_source_roll": self.source_roll,
                    "pitch_correction": float(self.pitch_correction_var.get()), "roll_correction": float(self.roll_correction_var.get()),
                    "cumulative_rotation": self.cumulative, "target_yaw": float(self.source_var.get()) + self.cumulative,
                    "accepted_by_user": accepted, "elapsed_seconds": elapsed, "last_inference_seconds": self.last_infer,
                    "diffusion_refined": self.was_refined,
                    "localized_anime_refined_side": self.localized_refined_side,
                    "same_character_frontal_reference_used": self.frontal_reference_used,
                    "same_character_frontal_reference": str(DEFAULT_FRONTAL_REFERENCE) if self.frontal_reference_used else None,
                    "same_character_frontal_reference_direct_candidate": self.frontal_reference_used,
                    "same_character_frontal_reference_guided_repair": self.reference_guided_used,
                    "frontal_reference_guidance_scale": float(self.reference_scale_var.get()),
                    "refiner_strength": float(self.strength_var.get()), "refiner_seed": int(self.seed_var.get()),
                    "crop_box": self.source_box, "saved_at": datetime.now().isoformat(timespec="seconds")}
        (RESULTS / f"{stem}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_var.set(f"已保存：{image_path.name}")
        messagebox.showinfo("保存成功", f"图像和实验参数已保存到：\n{RESULTS}")

    def _fit(self, image: np.ndarray, canvas: tk.Canvas):
        cw, ch = max(2, canvas.winfo_width()), max(2, canvas.winfo_height())
        h, w = image.shape[:2]; scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        return ImageTk.PhotoImage(Image.fromarray(image).resize((nw, nh), Image.Resampling.LANCZOS)), scale, (cw-nw)//2, (ch-nh)//2

    def _draw_images(self) -> None:
        if self.original is None: return
        self.left.delete("all"); self.right.delete("all")
        self.left_photo, self.left_scale, self.left_offset_x, self.left_offset_y = self._fit(self.original, self.left)
        self.left.create_image(self.left_offset_x, self.left_offset_y, image=self.left_photo, anchor="nw")
        if self.source_box:
            x0,y0,x1,y1=self.source_box; s=self.left_scale; ox=self.left_offset_x; oy=self.left_offset_y
            self.left.create_rectangle(ox+x0*s,oy+y0*s,ox+x1*s,oy+y1*s,outline="#00e5ff",width=3,tags="crop")
        shown = self.result if self.result is not None else self.original
        self.right_photo, _, ox, oy = self._fit(shown, self.right)
        self.right.create_image(ox, oy, image=self.right_photo, anchor="nw")

    def _drag_start(self, e) -> None: self.drag_start = (e.x, e.y)
    def _drag_move(self, e) -> None:
        self.left.delete("draft"); x,y=self.drag_start
        self.left.create_rectangle(x,y,e.x,e.y,outline="#ffcc00",width=2,tags="draft")
    def _drag_end(self, e) -> None:
        if self.original is None: return
        sx,sy=self.drag_start; ox,oy=self.left_offset_x,self.left_offset_y; scale=self.left_scale
        x0,x1=sorted(((sx-ox)/scale,(e.x-ox)/scale)); y0,y1=sorted(((sy-oy)/scale,(e.y-oy)/scale))
        side=min(x1-x0,y1-y0)
        h,w=self.original.shape[:2]; x0=max(0,min(w-side,x0)); y0=max(0,min(h-side,y0))
        if side >= 32: self.source_box=(int(x0),int(y0),int(x0+side),int(y0+side)); self._draw_images()


if __name__ == "__main__":
    PortraitApp().mainloop()
