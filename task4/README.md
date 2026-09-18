# Task 4：CPU 深度模型交互式侧脸转正脸

本实现使用 FasterLivePortrait 导出的 LivePortrait ONNX 权重。动漫人物使用 Human 模型，小猫使用 Animal 模型；推理明确限定为 `CPUExecutionProvider`，模型下载完成后可完全断网运行。这里采用标准 ONNX 图，而不是包含 TensorRT 自定义 `GridSample3D` 算子的 `-fix/v1.1` 图，保证 Windows CPU ONNX Runtime 可加载。

## 一次性准备

在 PowerShell 中执行：

```powershell
conda activate interactive-segmentation
python -m pip install -r task4/requirements-task4.txt
python task4/prepare_models.py --with-refiner
python task4/prepare_models.py --with-reference-adapter
```

角度模型约 1.08 GB，保存到 `task4/models/`；动漫局部修复使用 `Sanster/anything-4.0-inpainting`（`D:\task4-models/inpaint_anime`），小猫局部修复使用写实 `runwayml/stable-diffusion-inpainting`（`D:\task4-models/inpaint`）。这些权重不会提交到 Git。角度模型代码依据：[FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) 与 [LivePortrait](https://github.com/KlingAIResearch/LivePortrait)。

## 启动

```powershell
conda activate interactive-segmentation
python task4/task4.py
```

FLUX.2 Klein 的 CPU 服务与 GPU 服务可同时运行。CPU 服务使用 8192 端口；在另一个 PowerShell 窗口中启动它：

```powershell
conda activate task4-flux
./task4/launch_flux2_2.ps1
```

随后在界面中点击“FLUX.2 Klein 原图编辑生成正脸（CPU）”。CPU 版本会使用与 GPU 版本相同的模型和工作流，但速度会明显更慢；建议保留足够的系统内存和分页文件。

界面分为原图、结果、交互操作三栏。首先用角度按钮快速找到接近正脸的姿态，再点击“局部五官修复”。该步骤使用 inpainting，只处理双眼、鼻口和脸颊中央蒙版，并将结果羽化回填；头发、耳朵、衣服和背景不会被扩散结果覆盖。累计角度始终基于原图重新生成，不会把上一轮生成图再次送入角度模型。

- `模型估计原始 yaw` 是模型判断；极端侧脸或动漫图可能估计不准，可在“校准原始 yaw”手工修正。
- `pitch 校正比例` 与 `roll 校正比例` 控制仰俯和倾斜向 0° 收敛的程度；默认值为 0.55 / 0.70。设为 0 保留原姿态，设为 1 强制校正至 0°。
- “一步转到 0°”会按校准值生成正脸。
- LivePortrait 的快速预览只用于确定姿态；大侧脸必须再执行“局部五官修复”，不应直接把模糊预览当成最终结果。
- 动漫默认使用强度 `0.55`、种子 `2028`；小猫默认使用强度 `0.62`、种子 `2026`。太低无法修好远侧眼，太高会改变角色/猫的特征；可改变随机种子生成其他候选。
- 若拥有同一动漫角色的正脸图，可放在 `task4/data/face-正.png`，点击“显示同角色正脸参考候选（不融合）”。该候选图会独立显示和保存，不与 LivePortrait 结果叠加，以避免不同构图导致双脸重影；小猫不会使用此参考图。
- 完成 `--with-reference-adapter` 下载后，动漫“局部五官修复”会默认勾选“修复时参考同角色正脸图”。此时 `face-正.png` 会作为 IP-Adapter 图像条件进入扩散 inpainting，真正约束眼睛、鼻口和角色特征，而非像素叠加；默认引导强度为 `0.55`。小猫不会加载或使用该适配器。该图像编码器约 2.6 GB，CPU 修复会更慢。
- 原图上拖动鼠标可框选正方形头部，随后点“按当前框重新准备”。框应包含完整头发/双耳、眼睛、鼻口，减少背景。
- “满意”由用户人工确认，同时停止计时并保存；程序不会把自动生成等同于用户满意。
- 结果图和 JSON 参数/耗时记录写入 `task4/results/`，原图不会覆盖。

## 注意

LivePortrait Human 并非动漫专用模型，Animal 对大角度遮挡部分也只能做纹理形变，因此预览可能模糊。动漫局部 inpainting 负责重建缺失五官，小猫则使用独立的写实修复模型；扩散生成仍可能改变蒙版内的细节，最终必须由用户比较候选并确认。CPU 上局部修复可能需要数分钟，但不会调用 GPU 或远程推理服务。角度正负方向受模型定义影响，若首次方向相反，撤销后点击另一方向即可。
