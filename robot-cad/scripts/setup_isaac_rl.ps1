param(
    [string]$IsaacPython = "C:\isaacsim\python.bat"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$RequirementsPath = Join-Path $ProjectRoot "simulation\isaac\rl\requirements.txt"

if (-not (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) {
    throw "Isaac Sim Python launcher was not found: $IsaacPython"
}
if (-not (Test-Path -LiteralPath $RequirementsPath -PathType Leaf)) {
    throw "RL requirements file was not found: $RequirementsPath"
}

& $IsaacPython -m pip install `
    torch==2.10.0 `
    torchvision==0.25.0 `
    --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    throw "CUDA PyTorch installation failed with exit code $LASTEXITCODE"
}

& $IsaacPython -m pip install -r $RequirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "RL dependency installation failed with exit code $LASTEXITCODE"
}

& $IsaacPython -c @"
import gymnasium
import stable_baselines3
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"gymnasium={gymnasium.__version__}")
print(f"stable_baselines3={stable_baselines3.__version__}")
"@
if ($LASTEXITCODE -ne 0) {
    throw "RL runtime import check failed with exit code $LASTEXITCODE"
}
