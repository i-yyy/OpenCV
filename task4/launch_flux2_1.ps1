# Independent local GPU service for FLUX.2 Klein 4B image editing.
# It does not replace the FLUX.1 services on ports 8188/8189.
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

conda run --no-capture-output -n task4-flux python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "task4-flux has no CUDA PyTorch. Start the existing GPU setup first."
}

Set-Location $comfyRoot
# A separate process, port and SQLite database let it coexist with FLUX.1.
conda run --no-capture-output -n task4-flux python main.py `
    --listen 127.0.0.1 --port 8191 --lowvram --disable-mmap --disable-auto-launch `
    --database-url sqlite:///D:/task4-models/flux2/comfyui_gpu.db
