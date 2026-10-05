# Real-dataset campaigns for Work/server execution

This document describes the reproducible experiment campaigns prepared for the three
thesis datasets:

- MIRAGE-GenAI-2025;
- CESNET-QUICEXT-25 (monthly Zenodo top-50 registrable-domain task);
- CICIoT2023 (binary, family and fine-grained tasks).

All core profiles use 30 FL rounds and fixed preprocessing/client assignments.

## 1. Work/server concurrency

Do **not** pass `--max-concurrent-clients` to `eiffel.campaign_runner` and do not
pass `-MaxConcurrentClients` to `run.cmd` in Work/server mode.

Eiffel will ask Ray for the maximum safe client concurrency allowed by the available
resources. The campaign runner executes complete FL runs sequentially because each run
already consumes the available compute; parallel full experiments would generally
oversubscribe CPU/GPU/RAM.

Use a local cap only on constrained machines, for example:

```powershell
.\run.cmd -Suite mirage -MaxConcurrentClients 2
```

## 2. Required preprocessed data

Work only needs the processed artifacts, not the original raw captures/CSVs.

Expected roots:

```text
data/mirage/tasks/app_3class/
data/cesnet/
data/ciciot/
```

CESNET and CICIoT must contain `train_scaled.parquet`,
`validation_scaled.parquet`, `test_scaled.parquet`, `manifest.json`, and
`clients/client_00.parquet` ... `client_09.parquet`.

MIRAGE uses the existing preprocessing-kit layout and in particular
`train_scaled_with_clients.parquet` and `test_scaled.parquet`.

## 3. Core 30-round attack suites

Validate without training:

```powershell
.\run.cmd -Suite mirage -DryRun
.\run.cmd -Suite cesnet -DryRun
.\run.cmd -Suite ciciot-binary -DryRun
.\run.cmd -Suite ciciot-family -DryRun
.\run.cmd -Suite ciciot-fine -DryRun
```

Run:

```powershell
.\run.cmd -Suite mirage
.\run.cmd -Suite cesnet
.\run.cmd -Suite ciciot-binary
.\run.cmd -Suite ciciot-family
.\run.cmd -Suite ciciot-fine
```

Each suite contains:

1. clean;
2. targeted label flip;
3. Sign Flip;
4. Model Scaling;
5. Gaussian Noise;
6. LIE;
7. Gradient Mimicry;
8. Colluding Sign Flip;
9. Min-Max;
10. Min-Sum;
11. Adaptive Stealth;
12. Heterogeneity-Aware Mimicry;
13. Targeted Family/Class Poisoning.

## 4. Multi-seed replication

Example: replicate MIRAGE Sign Flip with five seeds.

```powershell
.\.venv\Scripts\python.exe -m eiffel.campaign_runner `
  experiments\toml\mirage_app3_quick_sign_flip.toml `
  --output-root outputs\mirage\campaigns\sign_flip_5seeds `
  --seeds 2026,2027,2028,2029,2030
```

The same pattern works for CESNET/CICIoT by changing the base profile.

## 5. Exact malicious-client study

The framework supports true logical client IDs from the preprocessing `client_id`
assignment.

All 45 two-attacker combinations on MIRAGE:

```powershell
.\.venv\Scripts\python.exe -m eiffel.campaign_runner `
  experiments\toml\mirage_app3_quick_sign_flip.toml `
  --output-root outputs\mirage\campaigns\sign_flip_all_pairs `
  --all-malicious-combinations 2
```

All 45 pairs and three seeds (135 runs):

```powershell
.\.venv\Scripts\python.exe -m eiffel.campaign_runner `
  experiments\toml\mirage_app3_quick_targeted_family_poisoning.toml `
  --output-root outputs\mirage\campaigns\targeted_pairs_3seeds `
  --all-malicious-combinations 2 `
  --seeds 2026,2027,2028
```

The same study is supported for CESNET and every CICIoT task.

Single compromised-client study:

```powershell
.\.venv\Scripts\python.exe -m eiffel.campaign_runner `
  experiments\toml\ciciot_family_sign_flip.toml `
  --output-root outputs\ciciot\family\campaigns\single_attacker `
  --all-malicious-combinations 1
```

## 6. Malicious-fraction sweep

Do not combine this sweep with `--all-malicious-combinations`.

```powershell
.\.venv\Scripts\python.exe -m eiffel.campaign_runner `
  experiments\toml\ciciot_family_sign_flip.toml `
  --output-root outputs\ciciot\family\campaigns\malicious_fraction `
  --set attack.malicious_fraction=0.1,0.2,0.3,0.4
```

## 7. Attack-strength sweeps

Sign Flip:

```powershell
--set attack.strength=1.0,2.0,3.0,5.0
```

Model Scaling:

```powershell
--set attack.strength=2.0,5.0,10.0,12.0
```

Gaussian Noise:

```powershell
--set attack.noise_std=0.5,1.0,5.0,10.0,20.0
```

LIE:

```powershell
--set attack.lie_z=0.5,1.0,1.5,2.0
```

Gradient Mimicry:

```powershell
--set attack.mimicry_lambda=0.5,0.75,0.85,0.95
```

Adaptive Stealth:

```powershell
--set attack.adaptive.max_strength=2.0,4.0,8.0
```

Targeted Family/Class Poisoning:

```powershell
--set attack.targeted.target_amplification=1.5,2.5,4.0
```

Multiple `--set` options produce a Cartesian product.

## 8. Temporal schedules

Example:

```powershell
.\.venv\Scripts\python.exe -m eiffel.campaign_runner `
  experiments\toml\mirage_app3_quick_sign_flip.toml `
  --output-root outputs\mirage\campaigns\sign_flip_schedules `
  --set attack.schedule.type=continuous,late,on_off,gradual
```

Defaults are deterministic:

- `late`: starts halfway through the configured rounds if no start is supplied;
- `on_off`: one active / one inactive round unless overridden;
- `gradual`: ramps from the configured/default initial to final multiplier.

For thesis reporting, create an explicit TOML profile when a non-default schedule is a
final experimental condition.

## 9. Target-class sweeps

The targeted update-space attack accepts both semantic names and generic
`class_id:N` selectors.

MIRAGE semantic targets can be `ChatGPT`, `Copilot`, or `Gemini`.

CICIoT family targets include `DDoS`, `DoS`, `Mirai`, `Recon`,
`Spoofing`, `Web`, `Bruteforce`, and `Benign`.

CESNET's classes are selected dynamically from the train period, so class IDs are the
portable target representation:

```powershell
.\.venv\Scripts\python.exe -m eiffel.campaign_runner `
  experiments\toml\cesnet_top50_targeted_family_poisoning.toml `
  --output-root outputs\cesnet\top50\campaigns\target_ids `
  --set attack.targeted.target_family=class_id:0,class_id:1,class_id:2
```

The CESNET `manifest.json` records the class-id/name mapping.

## 10. Combined scientific grids

Example: 3 seeds × 45 attacker pairs × 3 Sign Flip strengths = 405 runs.

```powershell
.\.venv\Scripts\python.exe -m eiffel.campaign_runner `
  experiments\toml\mirage_app3_quick_sign_flip.toml `
  --output-root outputs\mirage\campaigns\pairs_seed_strength `
  --all-malicious-combinations 2 `
  --seeds 2026,2027,2028 `
  --set attack.strength=1.5,3.0,5.0
```

Always run `--dry-run` first for large grids. The runner writes
`campaign_manifest.csv` containing variant ID, seed, malicious IDs, output path,
return code, and exact Eiffel command.

## 11. Analysis

Core suites:

```powershell
.\run.cmd -Suite mirage -AnalyzeDetailed
.\run.cmd -Suite cesnet -AnalyzeDetailed
.\run.cmd -Suite ciciot-family -AnalyzeDetailed
.\run.cmd -Suite ciciot-all -AnalyzeDetailed
```

Campaign directories can be analyzed directly:

```powershell
.\run.cmd -AnalyzeDetailed `
  -RunsRoot "outputs\mirage\campaigns\sign_flip_all_pairs" `
  -AnalysisOutput "analysis-results\mirage\campaigns\sign_flip_all_pairs"
```

Multiclass analyses include accuracy, macro-F1, macro class recall, worst-class
recall, per-class precision/recall/F1, update geometry, and the stored inference
probes. Binary CICIoT additionally retains per-attack-family recall/miss-rate.
