# Waits until Study T's 170 runs are complete, then launches Study U (run_u.py, 8 workers) and appends the launch to RUN_LOG_U.md.
$u = Split-Path -Parent $MyInvocation.MyCommand.Path
$t = Join-Path (Split-Path -Parent $u) "study_t\results_t"
while (@(Get-ChildItem -Path $t -Filter *.json -ErrorAction SilentlyContinue).Count -lt 170) { Start-Sleep -Seconds 60 }
$proc = Start-Process -FilePath python -ArgumentList "run_u.py", "--workers", "8" -WorkingDirectory $u `
    -RedirectStandardOutput (Join-Path $u "run_full.log") -RedirectStandardError (Join-Path $u "run_full.err") -WindowStyle Hidden -PassThru
$stamp = Get-Date -Format "HH:mm"
Add-Content -Path (Join-Path $u "RUN_LOG_U.md") -Value "- **$stamp.** Study T complete; full run launched automatically by launch_u_after_t.ps1 (PID $($proc.Id), ``run_full.log``): 80 runs, 8 workers." -Encoding UTF8
