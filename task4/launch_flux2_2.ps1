# Independent local CPU service for FLUX.2 Klein image editing.
# Port 8192 lets it coexist with the GPU FLUX.2 service on port 8191.
$ErrorActionPreference = "Stop"
$comfyRoot = Join-Path $PSScriptRoot "vendor\ComfyUI"

$required = @(
    "D:\task4-models\flux2\transformer\flux-2-klein-4b-fp8.safetensors",
    "D:\task4-models\flux2\comfy\split_files\text_encoders\qwen_3_4b.safetensors",
    "D:\task4-models\flux2\comfy\split_files\vae\flux2-vae.safetensors"
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing FLUX.2 model file: $path"
    }
}

Set-Location $comfyRoot
# Do not pass --disable-mmap: the 8 GB Qwen text encoder should remain
# memory-mapped on CPU instead of being copied into RAM in one allocation.
conda run --no-capture-output -n task4-flux python main.py `
    --cpu --cpu-vae --listen 127.0.0.1 --port 8192 --disable-auto-launch `
    --database-url sqlite:///D:/task4-models/flux2/comfyui_cpu.db
