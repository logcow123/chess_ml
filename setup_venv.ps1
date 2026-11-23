# Creates .venv for training on Windows with an NVIDIA GPU (RTX 3060).
# Run it by double-clicking setup_venv.bat, or from PowerShell in the repo root:
#     powershell -ExecutionPolicy Bypass -File setup_venv.ps1
# Safe to re-run: it reuses an existing .venv and just tops up packages.
# Everything printed is also written to setup_venv.log.
#
# Why these versions: maia2 (in requirements.txt) pins torch==2.8.0, numpy==2.1.3,
# pandas==2.2.3 and pyzstd==0.15.9. pyzstd 0.15.9 has no Windows wheel for Python 3.13+,
# so the venv uses Python 3.12 (or 3.11). torch 2.8.0 is installed from PyTorch's CUDA
# index first, so pip sees the pin as already satisfied and never swaps in a CPU build.

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$Log = Join-Path $PSScriptRoot 'setup_venv.log'
"setup_venv started $(Get-Date -Format s)" | Out-File $Log -Encoding utf8

function Say([string]$m, [string]$color = 'Cyan') {
    Write-Host "`n=== $m" -ForegroundColor $color
    "`n=== $m" | Out-File $Log -Append -Encoding utf8
}
function Run([string]$exe, [string[]]$argv) {
    & $exe @argv 2>&1 | ForEach-Object { $l = "$_"; Write-Host $l; $l | Out-File $Log -Append -Encoding utf8 }
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $exe $($argv -join ' ')" }
}

function Find-Python {
    # Prefer the py launcher (python.org installs), newest supported version first.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in '3.12', '3.11') {
            $p = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { return "$p".Trim() }
        }
    }
    foreach ($name in 'python', 'python3') {
        $c = Get-Command $name -ErrorAction SilentlyContinue
        if ($c -and $c.Source -notlike '*WindowsApps*') {   # skip the Microsoft Store stub
            $p = & $c.Source -c "import sys; print(sys.executable if sys.version_info[:2] in ((3,11),(3,12)) else '')" 2>$null
            if ($p -and "$p".Trim()) { return "$p".Trim() }
        }
    }
    return $null
}

try {
    $venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

    # 1. Python + venv ---------------------------------------------------------------
    $reuse = $false
    if (Test-Path $venvPy) {
        $ver = & $venvPy -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($ver -in '3.11', '3.12') { $reuse = $true; Say "Reusing existing .venv (Python $ver)" }
        else {
            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            Say "Existing .venv uses Python $ver, which maia2's pins don't support; moving it to .venv-old-$stamp" 'Yellow'
            Rename-Item '.venv' ".venv-old-$stamp"
        }
    }
    if (-not $reuse) {
        $py = Find-Python
        if (-not $py) {
            Say "Python 3.12 (or 3.11) not found. Install it, then run this again:" 'Red'
            Say "  winget install -e --id Python.Python.3.12     (or python.org/downloads, tick 'Add to PATH')" 'Red'
            exit 1
        }
        Say "Creating .venv with $py"
        Run $py @('-m', 'venv', '.venv')
    }
    Run $venvPy @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel')

    # 2. Pick the CUDA build from the driver -------------------------------------------
    $cuda = 'cu126'   # works with NVIDIA driver 560+
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $smi = (& nvidia-smi 2>&1) -join "`n"
        $smi | Out-File $Log -Append -Encoding utf8
        if ($smi -match 'CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)') {
            $drv = [int]$matches[1] * 10 + [int]$matches[2]
            if ($drv -ge 128) { $cuda = 'cu128' }
            Say "Driver supports CUDA $($matches[1]).$($matches[2]) -> installing torch 2.8.0+$cuda"
            if ($drv -lt 126) {
                Say "Your NVIDIA driver is too old for these builds. Update it (GeForce Experience / nvidia.com) and re-run." 'Yellow'
            }
        }
    } else {
        Say "nvidia-smi not found: is the NVIDIA driver installed? Installing torch+$cuda anyway." 'Yellow'
    }

    # 3. Packages -------------------------------------------------------------------------
    Say "Installing PyTorch 2.8.0 ($cuda) - about 3 GB, this is the slow part"
    Run $venvPy @('-m', 'pip', 'install', 'torch==2.8.0', '--index-url', "https://download.pytorch.org/whl/$cuda")

    Say "Installing requirements.txt"
    Run $venvPy @('-m', 'pip', 'install', '-r', 'requirements.txt')

    Say "Installing the chesspred package (editable)"
    Run $venvPy @('-m', 'pip', 'install', '-e', '.')

    # 4. Check it -------------------------------------------------------------------------
    Say "Checking the GPU and imports"
    $check = @'
import torch, chess, numpy, maia2, chesspred
from chesspred.models import build_model
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| numpy", numpy.__version__)
assert torch.version.cuda, "torch is a CPU build"
assert torch.cuda.is_available(), "CUDA not available - check the NVIDIA driver"
print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
m = build_model({"type": "resnet", "blocks": 6, "channels": 64}).cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(torch.randn(8, 18, 8, 8, device="cuda"), torch.full((8, 2), -1, device="cuda"))
print("ResNet 6x64 forward pass on GPU OK:", tuple(out.shape))
'@
    $checkFile = Join-Path $env:TEMP 'chesspred_venv_check.py'   # a file, not -c: PowerShell 5.1 mangles quotes in native args
    Set-Content -Path $checkFile -Value $check -Encoding ascii
    Run $venvPy @($checkFile)

    Say "Running the test suite"
    Run $venvPy @('-m', 'pytest', 'tests', '-q')

    Say "DONE - .venv is ready." 'Green'
    Say "Activate it with:  .\.venv\Scripts\Activate.ps1   then e.g.  python -m chesspred.train configs/resnet_6x64_lichess.yaml --overfit 64" 'Green'
    "SETUP_RESULT=OK" | Out-File $Log -Append -Encoding utf8
}
catch {
    Say "FAILED: $_" 'Red'
    "SETUP_RESULT=FAILED" | Out-File $Log -Append -Encoding utf8
    exit 1
}
