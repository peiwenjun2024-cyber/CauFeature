# RQ1 project input versions

The project CSVs and their fixed splits are identified by SHA-256 in
`rq1-inputs.csv`. Prepare inputs in a working directory using the
[public data commands](../PUBLIC_DATA.md). The dataset preparation seed
below applies before the fixed RQ1 seeds S01–S30.

## FeatureX archive

The FeatureX project distributes `Dataset.rar` at commit
`dd7d48ce3cce58d3bca98322244f09cfa3303cc5`. The archive has SHA-256
`3c1f28f2cc47c59ac61957ad93edbeba38dbc2f7a3a6949d648fedade40c9cee`.
The same archive bytes are preserved in the project's AutoDL vendor copies.
Only the named members below are needed. [FeatureX dataset page](https://uzhangyang.github.io/research/featurex.html)
[Pinned archive](https://raw.githubusercontent.com/aaaa-res/FeatureX/dd7d48ce3cce58d3bca98322244f09cfa3303cc5/Dataset.rar)

| Project dataset | FeatureX member | Preparation |
| --- | --- | --- |
| BrainMethod | `Dataset/BM.csv`, SHA-256 `e91eeef45cc149abf02aa0aaa47b11beea11635d3a08fe82424a6dc2e889a661` | Drop `NAME`, keep all 79,403 rows and their order, and write CRLF CSV. |
| CustomerChurn | `Dataset/churn.csv`, SHA-256 `3a3d44cdfb4e4242d7dde8da9e38ab617bcca42566c0ab6442b35da528bf865b` | Drop `ID`, keep all 5,227 rows and their order, and write CRLF CSV. |
| HeartDisease | `Dataset/heart.csv` | Use unchanged. |
| Income | `Dataset/adult_income.csv` | This member is the processed project input. The UCI conversion below rebuilds it from the upstream inputs. |
| SmokeAlarm | `Dataset/smoke_detection_iot.csv` | Apply the SmokeAlarm normalization below. |

The BrainMethod and CustomerChurn transformations require no sampling seed.
The original conversion commands that removed the identifier columns remain
unlocated. The named FeatureX member produces the archived CustomerChurn
input after removing `ID`. The IBM 7,043-row Telco release has different
row contents and counts.

## DiabetesHealth_sample10

Use the 253,680-row binary BRFSS release. Both the UCI CSV with `ID` and
the `diabetes_binary_health_indicators_BRFSS2015.csv` member of the
Alex Teboul release have the required row order. The UCI `data.csv` used
for reconstruction has SHA-256
`9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964`.
[UCI release](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
[Alex Teboul release](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

The archived full project file is `DiabetesHealth.csv`. Move
`Diabetes_binary` to the last column, drop `ID` if present, and remove the
decimal `.0` from integer-valued fields. The archived 10% derivative was
saved on 2026-09-24. Reproduce it by taking the first 10% of the NumPy
`RandomState(20260924).permutation` in the returned order, without
replacement. Serialize the 25,368 rows as UTF-8 CSV with LF line endings.
`scripts/prepare_public_data.py` implements the reconstructed procedure.
The original sampling script or invocation remains unlocated.

## HeartDisease

Use the 1,025-row `heart.csv` derivative. That file is already present under
`FeatureX-CauFeature/Dataset/` in the pre-migration project snapshot. The
available public mirror is pinned to commit
`752356b7d7591990fcb76017c587c7b4f0a93f84`. Keep the bytes, row order,
duplicate records and CRLF line endings. There is no preparation sampling
step. The exact historical download URL or Kaggle version remains
unconfirmed. [Pinned CSV](https://raw.githubusercontent.com/vngak-ai/sit307-11.1HD-heart-stacking/752356b7d7591990fcb76017c587c7b4f0a93f84/data/heart.csv)
[Kaggle derivative cited by the mirror](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## Income

Use both `adult.data` and `adult.test` from the UCI Adult ZIP with SHA-256
`7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`.
The matching pre-migration project version is `Dataset/adult_income.csv`.
Strip field whitespace, skip headers and invalid rows, drop rows containing
a literal `?`, and remove the test-target suffix `.`. Skip the first three
remaining rows of `adult.data` and the first remaining row of `adult.test`.
Concatenate the two files in that order and write CRLF CSV with the project
header. This row transformation is deterministic. The original conversion
script and reason for the four initial-row omissions remain unconfirmed.
The later `convert_adult()` audit script takes only `adult.data` and creates
a separate robustness input. [UCI Adult archive](https://archive.ics.uci.edu/static/public/2/adult.zip)

## SmokeAlarm

The pre-migration download notes identify the Kaggle Smoke Detection
Dataset. Its archived `originaldata/smoke_detection_iot.csv` matches the
public mirror pinned to commit `3c9f0f89bbcec1aacd9e9857e0a9ec797b2b58e8`
with SHA-256
`7bbf5deaab4c94746b344cee7bfec493f69b7d624ea63f35fabcd3c4a4e6eb7d`.
The original author's `data/final.csv` at commit
`e1b9d33cbca5885e2bb68fe4467e6e6960eebf91` is also a supported input.
Drop `UTC` and the unnamed index if present, move `Fire Alarm` to the last
column, strip any terminal `.0` from data fields, and write CRLF CSV. All
62,630 records keep their original order. The normalization script used in
the old project remains unlocated. [Kaggle release](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset)
[Pinned Kaggle mirror](https://raw.githubusercontent.com/AltagiAbdallah/Smoke-Detection-Analysis/3c9f0f89bbcec1aacd9e9857e0a9ec797b2b58e8/smoke_detection_iot.csv)
[Original author release](https://github.com/Blatts01/ai-smokedetector/blob/e1b9d33cbca5885e2bb68fe4467e6e6960eebf91/data/final.csv)
