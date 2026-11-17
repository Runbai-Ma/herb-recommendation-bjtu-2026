# Herb Prescription Recommendation (BJTU Web Search & Recommendation 2026)

Top-5 herb recommendation for the TCM prescription competition (Web Search and
Recommendation Systems course project, Beijing Jiaotong University, Fall 2026).
Given a patient's symptoms, recommend 5 herbs out of 805 candidates.
Primary metric: macro-averaged **F1@5** over prescription records.

**Final leaderboard score: see the version table below** (local sequential
validation is calibrated against the platform with a stable gap of ~0.043).

## Data

Five CSV files (UTF-8 BOM, list fields are JSON integer arrays), provided on the
course platform. Place them under `data/`:

| file | columns | rows | meaning |
| --- | --- | ---: | --- |
| data/user_features.csv | user_id, symptom_ids | 33,765 | symptoms of every record |
| data/items.csv | item_id, herb_name | 805 | herb vocabulary |
| data/symptoms.csv | symptom_id, symptom_name | 390 | symptom vocabulary |
| data/train_interactions.csv | user_id, item_id | 175,943 | positive interactions of 27,012 training records |
| data/test.csv | user_id, symptom_ids | 6,753 | test records (train/test split by record, 8:2) |

`features/*.npz` contains pre-computed 512-d name vectors (`BAAI/bge-small-zh-v1.5`)
for the symptom and herb vocabularies, so no model download is needed.

## Key facts that shaped the method

- Records carry on average **1.48 symptoms** but **6.5 herbs** - the
  symptom-set -> prescription conditional distribution is the main signal.
- Train/test are split **by user_id order**, and the symptom->herb mapping
  **drifts along that order** (top-8 conditional herbs of frequent symptoms
  agree only 66% between early/late segments). Random-split validation
  overestimates platform score by ~0.065; all tuning below uses sequential
  block splits cross-checked at two cut points (80/20 and 75/25).
- Under drift, the exact symptom-combination hit feature is misleading:
  removing it improved every member by +0.009 (exp15).

## Final recipe (v6)

10 MLP members, logits averaged (pure ensemble, no signal fusion):

- input per member: symptom ID embedding (64d) + mean-pooled BGE symptom-name
  vectors (512d) + statistical priors (conditional probability 805d + log
  popularity 805d; **no** combination-hit feature, **no** semantic feature);
- two-layer MLP (256d, BN + dropout) -> 805 logits;
- members: 5x lr5e-4/dropout0.1 + 3x lr5e-4/0.2 + 2x lr1e-3/0.1, 15 epochs on
  full data (median of sequential early-stop), seeds 2026/7/42/123/777/314/159/265/358/979.

Local sequential validation: **0.2335** (80/20) and **0.2292** (75/25), vs 0.17427
for the provided two-tower baseline.

## Version history (local sequential / platform)

| version | recipe | local | platform |
| --- | --- | ---: | ---: |
| baseline (two-tower, dim=32) | provided | ~0.174 | 0.17427 |
| v1 | tuned on random split (lesson: miscalibrated) | 0.2461 | 0.18145 |
| v2 | sequential-block protocol + shrunk combination statistics | 0.2258 | 0.18341 |
| v3 | + 5-member MLP ensemble, lr5e-4 | 0.2277 | 0.18484 |
| v4 | + semantic-name feature (platform-negative, reverted) | 0.2291 | 0.18255 |
| v5 | no semantics + 10 members | 0.2285 | 0.18505 |
| **v6** | **drop combination-hit feature, pure ensemble** | **0.2335** | see platform |

## Reproduce

```bash
conda create -n herb2026 python=3.12.2 pip -y
conda activate herb2026
python -m pip install numpy==1.26.4 torch==2.5.1 --index-url https://pypi.org/simple

python final_pipeline.py   # 10 members, ~6 min CPU, writes result_final/submission.csv
python baseline.py         # provided two-tower baseline for reference
```

`final_pipeline.py` defaults are the v6 configuration
(`--use-combo 0 --sem-feat 0 --w-stat 0 --w-model 1 --w-cond 0`).
Submission format: header `user_id,item_ids`, one row per test record with 5
distinct herb IDs, e.g. `27012,"[22,71,50,197,75]"`.

## Experiments

`experiments/exp1-16` reproduce the full study: statistical baselines,
sequential-drift diagnostics, estimator variants, hyper-parameter grids,
ensembling, and the cross-split confirmations. Scripts read data from `data/`
and write artifacts under `experiments/` (gitignored).

## Negative results (ablation summary)

- random-split tuning, semantic-name features, kNN records, IDF weighting,
  herb co-occurrence re-ranking, time-decayed statistics, symptom-neighbor
  expansion, wider/deeper towers, pos_weight, >10 ensemble members,
  per-group fusion weights - all failed to help (see `experiments/`).
- the combination-hit feature is a *negative* signal under drift: the clearest
  single finding of this study.

## License

MIT
