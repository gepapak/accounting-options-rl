# Waits until Study P's 90 runs are complete, then launches Study Q (run_q.py, 7 workers) and appends the launch to RUN_LOG_Q.md.
$q = Split-Path -Parent $MyInvocation.MyCommand.Path
$p = Join-Path (Split-Path -Parent $q) "study_p\results_p\jobs"
while (@(Get-ChildItem -Path $p -Filter *.json -ErrorAction SilentlyContinue).Count -lt 90) { Start-Sleep -Seconds 60 }
$proc = Start-Process -FilePath python -ArgumentList "run_q.py", "--workers", "7" -WorkingDirectory $q `
    -RedirectStandardOutput (Join-Path $q "run_full.log") -RedirectStandardError (Join-Path $q "run_full.err") -WindowStyle Hidden -PassThru
$stamp = Get-Date -Format "HH:mm"
Add-Content -Path (Join-Path $q "RUN_LOG_Q.md") -Value "- **$stamp.** Study P complete; full run launched automatically by launch_after_p.ps1 (PID $($proc.Id), ``run_full.log``): 100 runs, 7 workers." -Encoding UTF8
