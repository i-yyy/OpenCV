# 交互式图像分割

这是一个基于 `Tkinter + OpenCV GrabCut` 的 CPU 离线交互式图像分割程序。

## 环境

Windows 上推荐使用下面这条已经验证过的命令：

```powershell
conda --no-plugins create --solver classic --override-channels -c defaults --repodata-fn current_repodata.json -y -n interactive-segmentation python=3.12 numpy pillow opencv
conda activate interactive-segmentation
```

也可以使用环境文件：

```powershell
conda env create -f environment.yml
conda activate interactive-segmentation
```

如果当前 Conda 在 Windows 上受插件影响报权限错误，或提示 libmamba solver 不可用，可以加 `--no-plugins` 和 classic solver：

```powershell
conda --no-plugins env create --solver classic -f environment.yml
conda activate interactive-segmentation
```

## 运行

```powershell
python interactive_segmentation.py
```

程序默认会读取 `data/LENA.jpg`，也可以点击“读取图像”选择其他图片。界面左侧画布显示实验结果图像，右侧上方同步显示分割出的前景图，右侧下方同步显示分割掩码。

## 使用建议

1. 选择“初始区域”和“长方形”，在目标周围画一个粗略区域。
2. 点击“运行分割”。
3. 用“前景”涂抹漏掉的目标区域，用“背景”涂抹误分割区域。
4. 多次点击“运行分割”，直到结果满意。
5. 点击“保存结果”，会输出：
   - `*_overlay.png`
   - `*_mask.png`
   - `*_foreground.png`

界面中会同时显示实验结果图像、分割前景和分割掩码；叠加结果图会显示操作时间、交互次数和标记次数。运行分割后，左侧绿色区域始终表示最近一次算法分割出的前景，不再继续显示初始矩形等前景标记；继续涂抹后需要再次点击“运行分割”来更新绿色前景。

操作时间按有效图像操作累计，只统计实际涂抹、橡皮擦修改、形状/多边形标记完成和运行分割的耗时；单纯切换工具、切换视图或刷新界面不会增加时间。

如果某次标记画错了，可以选择“橡皮擦”并在错误区域拖动，程序会清掉该区域的可视化标记，并把对应 GrabCut 掩码移回背景。长方形、圆、椭圆、多边形等大面积前景标记会作为“可能前景”处理；更精确的“确定前景/背景”建议用涂抹工具修正。

使用长方形、正方形、圆、椭圆、直线和正多边形时，拖拽鼠标可以超出图片显示范围，程序会自动把终点限制在图片边界内，方便框选贴近边缘的目标。

“分割后优化”可以在每次运行 GrabCut 后继续处理掩码：开运算去掉零散噪点，闭运算填补小洞，保留最大区域去掉远处误分割块，轮廓平滑让边缘更自然。多目标图片建议关闭“保留最大区域”。

如果上一笔标记画错了，可以在运行分割前点击“撤销标记”或按 `Ctrl+Z` 恢复到这一笔之前；涂抹、橡皮擦、直线、矩形、圆、椭圆和多边形都可以撤销。每次点击“运行分割”后，当前标记撤销记录会清空，表示本轮标记已经提交。

