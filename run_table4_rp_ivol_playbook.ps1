$ErrorActionPreference = "Stop"

# Run from this script's directory (repo root expected)
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Starting RP IdioVol playbook (step1 + RP Part1/2 + Gaussian/Exponential)..." -ForegroundColor Cyan
python .\run_table4_rp_ivol_playbook.py @args
$code = $LASTEXITCODE

Write-Host ""
Write-Host "Exit code: $code" -ForegroundColor $(if ($code -eq 0) { "Green" } else { "Red" })
Write-Host ""
Read-Host "Press Enter to close"

exit $code
