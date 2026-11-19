# Study XE runner. Run from this folder only after hash_prereg_xe.py has written the hashes.
# $vbt and $mtsim are the two virtual environments' python.exe paths.
param([Parameter(Mandatory = $true)][string]$vbt, [Parameter(Mandatory = $true)][string]$mtsim)
$env:TQDM_DISABLE = "1"
& $vbt -B hash_prereg_xe.py --verify; if ($LASTEXITCODE -ne 0) { throw "hashes do not match" }
& $vbt -B engines_pilot.py 2000 20261201 xe_census.csv *> xe_census_out.txt                         # A (and B in aggregate_xe.py)
& $mtsim -B census_mtsim.py 300 20261202 xe_census_mtsim.csv *> xe_census_mtsim_out.txt             # D, gym-mtsim
& $vbt -B reading_engines.py *> reading_xe_out.txt                                                 # D, one-query reading
& $vbt -B rl_engine_pilot.py --steps 500000 --seeds 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 --engines btpy bt --workers 8 --levels 0 200 --tag _xe5 --test-seed 20261203 --out results_xe *> run_xe5_out.txt   # C
& $vbt -B rl_engine_pilot.py --steps 500000 --seeds 201 202 203 204 205 206 207 208 209 210 211 212 --engines btpy bt --workers 8 --levels -1 0 1 --lmax 1 --mags crypto_mags.npy --magscale 2 --tag _xe7 --test-seed 20261204 --out results_xe *> run_xe7_out.txt   # E
& $vbt -B aggregate_xe.py *> aggregate_xe_out.txt
