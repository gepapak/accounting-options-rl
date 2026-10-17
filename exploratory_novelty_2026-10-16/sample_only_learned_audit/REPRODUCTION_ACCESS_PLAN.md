# Exact dependency and local review-bundle plan

16 October 2026. Inventory/provenance only; no new actor inference, market
download, financial endpoint, training or statistical selection is performed.
The ongoing experiment's frozen sources and inputs remain read-only.

## Size and the exact native source version

| Material | Files | Bytes | MiB |
|---|---:|---:|---:|
| Frozen native Python source | 34 | 1,767,602 | 1.686 |
| Investor policy ZIPs | 20 | 28,395,516 | 27.080 |
| Matching battery policy ZIPs | 20 | 14,847,136 | 14.159 |
| Saved training/runtime JSON configs | 20 | 249,694 | 0.238 |
| Source plus paired checkpoints/configs | 94 | 45,259,948 | 43.163 |
| Conservative context/history/forecast inventory | 30 | 131,804,117 | 125.698 |
| Actual runtime CSV dependency subset | 10 | 53,910,563 | 51.413 |

MiB means bytes divided by 1,048,576. Counts and bytes are source-derived, before
compression and before small manifests/documentation. A source/checkpoint review
bundle is therefore modest. An exact-data runtime bundle would add about51MiB,
if separately authorized; no such CSV is included in the current private bundle.

The existing public `engine/` copy cannot substitute byte-for-byte for the
current frozen native source: 19 of34 modules match, 14 differ, and
`learning_wall_analysis.py` is absent. Differences include the environment,
configuration, observation builder, generator, forecast and evaluation code.
The local review bundle supplies all34 exact frozen modules and their hashes;
it does not silently replace the public audit copy or the running source.

## Runtime inputs: no eSett files

All six evaluation/settlement/liquidity files in this audit trace to Energinet
and author-created physical transformations. The two evaluation trajectories
contain observed day-ahead prices and ERA5-based wind, with synthetic
solar/hydro/load. DK2 changes the price/settlement/liquidity region while keeping
the physical trajectory shared. The two settlement and two liquidity files are
aligned derivatives of Energi Data Service APIs. The native builders do not
query eSett. The project's nonredistribution policy for raw eSett records belongs
to other studies; it is not an input-access restriction for these ten files.

The initial30-file manifest deliberately inventories the full local normalization
history directory. It is a conservative superset. Native
`evaluation.py:_prime_eval_rolling_past_from_history` starts from the newest
numbered histories and stops once its365-day trailing window is covered. Actual
phase metadata lists only `history_019.csv` and `history_020.csv` (2024-H1/H2).
Those files match `training_dataset_ffill/scenario_018.csv` and
`scenario_019.csv` in the native history manifest. The older19 histories,
including the frozen2014 bootstrap, and `history_manifest.csv` are not read by
the current audit. Reproducing the original RL training is a different, larger
dependency question; replaying its frozen actors needs only this trailing pair.

The ten operational CSVs are:

- `evaluation_dataset_ffill/unseendata.csv` and `unseendata_v2.csv`;
- the corresponding two `_settlement_real_v2.csv` files;
- the corresponding two `_liquidity_volume_real_v1.csv` files;
- `rolling_past_history_dataset_ffill/history_019.csv` and `history_020.csv`;
- the two episode20 original/v2 `precomputed_forecasts_*.csv` sources listed in
  the public manifest.

The last two are authors' ANN forecasts, not a purchased or third-party forecast
feed. They store predicted prices, directional/uncertainty/quality features and
latent coordinates. Native maturity signs are separately randomized by the audit;
these forecast sources remain the originally trained actor's observed context.
Existing exact features can be supplied as versioned input artifacts if
authorized. Regenerating them instead needs the frozen episode20 ANN bank and
the declared settlement-based forecast-input builder. Its five model/scaler/
metadata/history files exist and total1,950,415bytes (1.860MiB). That model bank
is not loaded by audit replay when the two cached CSVs are supplied, and it is
not copied into the current source/RL-checkpoint-only private bundle.

## License and attribution evidence

The repository's `THIRD_PARTY_LICENSES.md` identifies the authors' engine code
as MIT and the Energinet data as CC BY4.0. The official
[Energi Data Service terms](https://www.energidataservice.dk/terms-and-conditions)
permit copying, modification and redistribution with attribution and an
indication of changes. A suitable attribution is: Source: Energinet
(www.energidataservice.dk); aligned/resampled and transformed by the authors;
Energinet does not endorse this work.

The wind conversion source records ERA5 100m u/v components from Copernicus CDS,
at latitude55.5, longitude7.48, derived from the documented request area
[55.6,7.48,55.5,7.58]. The official
[ERA5 single-levels catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
identifies a CC-BY licence and DOI10.24381/cds.adbb2d47. The downloadable original
point series and the paper's wind-power/interpolation transformations require
the appropriate Copernicus/ERA5 attribution. A CDS account and licence acceptance
are needed for API download. The conversion script includes the request schema;
credentials are external and are not part of this bundle.

The synthetic physical trajectories and forecast outputs are author-created
derivatives. No third-party proprietary forecast provider or eSett raw input is
identified in this audit's native reconstruction chain. The repository's current
license files explicitly cover source, documentation/results and specified
derived tables; they do not explicitly name the policy ZIPs or this newly
packaged forecast/data bundle. A public release should assign an explicit author
license to those artifacts and include the Energinet/Copernicus attributions.
This is a concrete release-labeling task, not a barrier to the authorized local
private review copy. No public distribution or push is performed here.

## Reconstruction route and remaining exactness gaps

These are existing scripts, not commands executed during this inventory:

1. `dataset generation/wind speed data/convert_wind_nc_to_excel.py`: documented
   CDS request and 100m wind conversion to10-minute series.
2. `dataset generation/datagenerationforMARLtraining2015-2024.py` and
   `datagenerationforevaluationyear2025.py`: Energinet Elspotprices, wind and
   synthetic physical trajectories/templates; the training generator also uses
   PyPSA. Preserve frozen physical templates if byte-exact reproduction is
   required. A currently seeded generator is not evidence that all previously
   frozen stochastic artifacts will be regenerated byte-for-byte.
3. `scripts/build_ffill_training_protocol.py`: forward-filled price/history
   transformation; `scripts/finalize_protocol_provenance.py` identifies the
   actual frozen artifacts and creates history020 from the final2024-H2 file.
4. `datagenerationforevaluationyear2025_unseendata_v2.py` and
   `scripts/complete_evaluation_window.py`: shared physical trajectory, DK2
   hourly price replacement, and completion through2025-09-30 23:50.
5. `scripts/build_real_settlement_protocol_v2.py`: API-root
   `https://api.energidataservice.dk/dataset`; RegulatingBalancePowerdata and
   ImbalancePrice composite, configured support clips and declared alignment.
   It can also construct the episode20 forecast input files.
6. `scripts/build_real_liquidity_volume_protocol_v1.py`: Energinet activation
   volumes, component summation,10-minute alignment and zero/missing handling.
7. `scripts/precompute_hourly_forecast_cache_v2.py --skip_training_episodes`:
   reconstruct the two evaluation caches from the frozen episode20 ANN bank;
   run separately with each original/v2 forecast input. It uses24 hourly past
   settlement observations plus current entry information, one-hour forecasts,
   and repeats each origin-hour forecast over its10-minute rows. Do not retrain
   the bank merely to reproduce the existing actor's features.

Live API revisions, physical-template provenance, software/CSV formatting and
floating-point forecast inference can prevent byte-identical regeneration. The
direct way to close exact replay is to archive the ten frozen operational CSVs
with hashes/attribution, or to provide authorized access to those exact versions.
A downloader plus a hash check should report a mismatch rather than overwrite
the expected reference. Source plus checkpoints alone does not close this gap.

The tested native runtime is Python3.10 with PyTorch2.8.0 and
Stable-Baselines3 2.7.0. The public manifest records the other observed package
versions. The existing `requirements-engine.txt` is useful but a fresh clean
installation has not been validated for this packet. Input reconstruction adds
requests/xarray/NetCDF/CDS tools and, if physical optimization or ANN regeneration
is needed, PyPSA/solver and the saved model's TensorFlow/Keras stack. Shipping the
exact ten CSVs avoids retraining or physical-data regeneration during audit replay.

## Local review artifact

The authorized review directory is `review_bundle_private/native_review_v1/`,
gitignored and uncompressed. It contains exact native source and paired RL
checkpoint/config files, selected reconstruction SOURCE scripts, the audit's
hash manifests, license/access notes and clean-layout instructions. It contains
no market/history/forecast CSV, no evaluation feature arrays, no statistical
phase traces and no fitted audit nuisance models. Original files remain
read-only and copy hashes are checked. Training configs retain their original
absolute paths to preserve bytes; this is a private review artifact, not an
anonymized submission or a public release.
