# Runs the README training sequence on the RTX 3060 (after setup_venv.bat and fetch_data.bat):
#   1. overfit check (64 positions must reach ~100% top-1, else stop: there's a bug)
#   2. pre-train the generic ResNet on Lichess   -> models/runs/resnet_6x64_lichess/
#   3. fine-tune it on your chess.com games      -> models/runs/resnet_6x64_finetune_whopizza1/
# Double-click train_all.bat. Safe to re-run: steps 2-3 use --resume, so a stopped run
# continues from its last finished epoch and a finished one is skipped.
# Keeps the PC awake while it runs (only while this window is open; no settings change).
# Everything printed is also written to train.log.

param([switch]$SkipOverfit, [switch]$SkipPretrain, [switch]$SkipFinetune)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$Log = Join-Path $PSScriptRoot 'train.log'
"train_all started $(Get-Date -Format s)" | Out-File $Log -Encoding utf8

$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

# Ask Windows not to sleep while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
try {
    Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
    [void][Win32.Power]::SetThreadExecutionState([uint32]2147483649)
} catch { }

function Say([string]$m, [string]$color = 'Cyan') {
    Write-Host "`n=== $m" -ForegroundColor $color
    "`n=== $m" | Out-File $Log -Append -Encoding utf8
}
function Py([string[]]$argv) {
    $script:last = New-Object System.Collections.Generic.List[string]
    & $script:Py @argv 2>&1 | ForEach-Object {
        $l = "$_"; Write-Host $l; $l | Out-File $Log -Append -Encoding utf8; $script:last.Add($l)
    }
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): python $($argv -join ' ')" }
}

try {
    $script:Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $script:Py)) { throw ".venv not found - run setup_venv.bat first" }
    $pre = 'configs\resnet_6x64_lichess.yaml'
    $ft = 'configs\resnet_6x64_finetune_whopizza1.yaml'

    if (-not $SkipOverfit) {
        Say "1/3 Overfit check: can the model memorise 64 positions?"
        Py @('-m', 'chesspred.train', $pre, '--overfit', '64')
        if (-not (@($script:last) -match '^PASS')) { throw "Overfit check did not pass - not starting the long run" }
    }
    if (-not $SkipPretrain) {
        Say "2/3 Pre-training on Lichess (the long one; resumable)"
        Py @('-m', 'chesspred.train', $pre, '--resume')
    }
    if (-not $SkipFinetune) {
        Say "3/3 Fine-tuning on your games"
        Py @('-m', 'chesspred.train', $ft, '--resume')
    }

    Say "DONE. Checkpoints and log.csv are in models\runs\<run name>\" 'Green'
    "TRAIN_RESULT=OK" | Out-File $Log -Append -Encoding utf8
}
catch {
    Say "FAILED: $_" 'Red'
    "TRAIN_RESULT=FAILED" | Out-File $Log -Append -Encoding utf8
    exit 1
}
finally {
    try { [void][Win32.Power]::SetThreadExecutionState([uint32]2147483648) } catch { }
}
