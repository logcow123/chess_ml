# Downloads and prepares the training data (run setup_venv.bat first).
# Run it by double-clicking fetch_data.bat. Safe to re-run: finished steps are skipped,
# and a half-finished Lichess download resumes shard by shard.
# Everything printed is also written to fetch_data.log.
#
#   1. Your chess.com games  -> data/raw/chesscom_whopizza1_<today>.jsonl
#      parsed (rapid, your moves only) -> data/processed/..._rapid_whopizza1-moves.parquet
#      pre-encoded (frozen split)      -> ... .npz   (what the fine-tune config reads)
#   2. Lichess 2025-09, rated blitz/rapid, both players 900-1400
#      -> data/raw/lichess/lichess_2025-09_blitz-rapid_900-1400.parquet
#      parsed (rapid only)  -> data/processed/lichess_2025-09_blitz-rapid_900-1400_rapid.parquet
#      pre-encoded          -> ... .npz   (what the pre-training config reads)

param(
    [string]$Username = 'whopizza1',
    [int]$LichessYear = 2025,
    [int]$LichessMonth = 9,
    [int]$LichessShards = 4,
    [switch]$SkipChessCom,
    [switch]$SkipLichess
)

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$Log = Join-Path $PSScriptRoot 'fetch_data.log'
"fetch_data started $(Get-Date -Format s)" | Out-File $Log -Encoding utf8

$env:PYTHONUNBUFFERED = '1'              # print lines as they happen, not at the end
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'  # progress bars turn the log into one giant line
$env:TQDM_DISABLE = '1'

function Say([string]$m, [string]$color = 'Cyan') {
    Write-Host "`n=== $m" -ForegroundColor $color
    "`n=== $m" | Out-File $Log -Append -Encoding utf8
}
function Py([string[]]$argv) {
    & $script:Py @argv 2>&1 | ForEach-Object { $l = "$_"; Write-Host $l; $l | Out-File $Log -Append -Encoding utf8 }
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): python $($argv -join ' ')" }
}

try {
    $script:Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $script:Py)) { throw ".venv not found - run setup_venv.bat first" }
    New-Item -ItemType Directory -Force -Path 'data\raw', 'data\processed' | Out-Null

    # 1. chess.com --------------------------------------------------------------------
    if (-not $SkipChessCom) {
        $user = $Username.ToLower()
        $today = Get-Date -Format 'yyyy-MM-dd'
        $raw = "data\raw\chesscom_${user}_$today.jsonl"
        $parsed = "data\processed\chesscom_${user}_${today}_rapid_${user}-moves.parquet"
        $npz = [IO.Path]::ChangeExtension($parsed, '.npz')

        if (Test-Path $raw) { Say "chess.com: $raw already downloaded" }
        else { Say "chess.com: downloading all of $Username's games"; Py @('-m', 'chesspred.data.download_chesscom', '--username', $user) }

        if (Test-Path $parsed) { Say "chess.com: $parsed already parsed" }
        else { Say "chess.com: parsing rapid games, keeping $Username's moves"; Py @('-m', 'chesspred.data.parse_pgn', $raw, '--player', $user) }

        if (Test-Path $npz) { Say "chess.com: $npz already encoded" }
        else { Say "chess.com: pre-encoding (uses the frozen split in data\splits)"; Py @('-m', 'chesspred.encoding', $parsed) }

        # Point the fine-tune config at the file just made (it names a dated file).
        $cfg = 'configs\resnet_6x64_finetune_whopizza1.yaml'
        $want = ($npz -replace '\\', '/')
        $text = Get-Content $cfg -Raw
        $new = [regex]::Replace($text, '(?m)^(\s*train:\s*)data/processed/chesscom_\S+\.npz', "`${1}$want")
        if ($new -ne $text) {
            Set-Content -Path $cfg -Value $new -NoNewline -Encoding ascii
            Say "Updated $cfg to train on $want"
        }
    }

    # 2. Lichess ------------------------------------------------------------------------
    if (-not $SkipLichess) {
        $tag = '{0}-{1:D2}' -f $LichessYear, $LichessMonth
        $raw = "data\raw\lichess\lichess_${tag}_blitz-rapid_900-1400.parquet"
        $parsed = "data\processed\lichess_${tag}_blitz-rapid_900-1400_rapid.parquet"
        $npz = [IO.Path]::ChangeExtension($parsed, '.npz')

        if (Test-Path $raw) { Say "Lichess: $raw already downloaded" }
        else {
            Say "Lichess: downloading $LichessShards shards of $tag and filtering them (the slow part)"
            Py @('-m', 'chesspred.data.download_lichess', '--year', "$LichessYear", '--month', "$LichessMonth", '--shards', "$LichessShards")
        }

        if (Test-Path $parsed) { Say "Lichess: $parsed already parsed" }
        else { Say "Lichess: parsing rapid games into positions"; Py @('-m', 'chesspred.data.parse_pgn', $raw) }

        if (Test-Path $npz) { Say "Lichess: $npz already encoded" }
        else { Say "Lichess: pre-encoding"; Py @('-m', 'chesspred.encoding', $parsed) }
    }

    Say "DONE - data is ready. Next: python -m chesspred.train configs/resnet_6x64_lichess.yaml --overfit 64" 'Green'
    "FETCH_RESULT=OK" | Out-File $Log -Append -Encoding utf8
}
catch {
    Say "FAILED: $_" 'Red'
    "FETCH_RESULT=FAILED" | Out-File $Log -Append -Encoding utf8
    exit 1
}
