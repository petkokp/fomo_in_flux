# Compact Results Summary (eval-only)

Artifacts:
- `pivot_delta_all_groups.csv` — ALL Δ vs ZS (rows=groups `method|tX|nY`, cols=datasets).
- `group_summary.csv` — per-group means/stdevs (eval-only).
- `cleveland_delta_by_dataset.png` — Cleveland dot plot (per-dataset mean Δ by method).
- `bar_mean_delta_by_method.png` — global mean Δ per method.
- `waterfall_<method>.png` — per-method Δ distribution (sorted contributions).
- `slopegraph_<method>.png` — ZS → method absolute accuracy per dataset.

## Table — Group summary (eval-only)
| method | tasks | n_samples | rows | mean_delta | mean_acc_final |
|---|---|---|---|---|---|
| ema | 1 | 1024 | 9.0 | -0.74 | 40.38 |
| ema | 1 | 2048 | 9.0 | -1.60 | 39.52 |
| ema | 2 | 1024 | 9.0 | -0.70 | 40.43 |
| ema | 2 | 2048 | 9.0 | -4.97 | 36.16 |
| ema | 3 | 1024 | 9.0 | -0.61 | 40.52 |
| ema | 3 | 2048 | 9.0 | -11.97 | 29.15 |
| ema | 4 | 1024 | 9.0 | -1.09 | 40.03 |
| ema | 4 | 2048 | 9.0 | -25.03 | 16.10 |
| ema | 5 | 1024 | 9.0 | -0.74 | 40.39 |
| ema | 5 | 2048 | 9.0 | -38.51 | 2.62 |
| finetune | 1 | 1024 | 9.0 | -1.42 | 39.71 |
| finetune | 1 | 2048 | 9.0 | -1.60 | 39.52 |
| finetune | 2 | 1024 | 9.0 | -0.70 | 40.42 |
| finetune | 2 | 2048 | 9.0 | -2.41 | 38.71 |
| finetune | 3 | 1024 | 9.0 | -0.68 | 40.44 |
| finetune | 3 | 2048 | 9.0 | -2.94 | 38.19 |
| finetune | 4 | 1024 | 9.0 | -0.60 | 40.52 |
| finetune | 4 | 2048 | 9.0 | -2.36 | 38.77 |
| finetune | 5 | 1024 | 9.0 | -0.78 | 40.35 |
| finetune | 5 | 2048 | 9.0 | -3.46 | 37.67 |
| tda | N.A. | N.A. | 9.0 | 6.36 | 47.48 |
| vte | N.A. | N.A. | 9.0 | 7.33 | 48.45 |


## Appendix — Global extremes
#### Top improvements (Δ vs ZS)

| method | tasks | n_samples | dataset | delta_vs_zs | acc_final | zs_baseline |
|---|---|---|---|---|---|---|
| tda | N.A. | N.A. | plantvillage | 45.59 | 76.36 | 30.78 |
| tda | N.A. | N.A. | imagenet_a | 29.89 | 55.80 | 25.91 |
| vte | N.A. | N.A. | imagenet_a | 22.32 | 48.23 | 25.91 |
| vte | N.A. | N.A. | imagenet_d | 15.07 | 56.26 | 41.19 |
| vte | N.A. | N.A. | imagenet | 13.09 | 79.24 | 66.15 |
| vte | N.A. | N.A. | imagenet_s | 11.16 | 64.27 | 53.11 |
| vte | N.A. | N.A. | imagenet_v2 | 10.75 | 66.99 | 56.24 |
| tda | N.A. | N.A. | imagenet_r | 9.38 | 84.32 | 74.93 |
| tda | N.A. | N.A. | clevr | 9.27 | 14.42 | 5.15 |
| vte | N.A. | N.A. | imagenet_r | 8.87 | 83.80 | 74.93 |


#### Biggest drops (Δ vs ZS)

| method | tasks | n_samples | dataset | delta_vs_zs | acc_final | zs_baseline |
|---|---|---|---|---|---|---|
| ema | 5 | 2048 | imagenet_r | -70.64 | 4.29 | 74.93 |
| ema | 5 | 2048 | imagenet | -64.43 | 1.72 | 66.15 |
| ema | 5 | 2048 | imagenet_v2 | -54.65 | 1.59 | 56.24 |
| ema | 5 | 2048 | imagenet_s | -52.12 | 0.99 | 53.11 |
| ema | 4 | 2048 | imagenet | -43.11 | 23.04 | 66.15 |
| ema | 4 | 2048 | imagenet_s | -40.86 | 12.26 | 53.11 |
| ema | 4 | 2048 | imagenet_r | -39.25 | 35.68 | 74.93 |
| ema | 5 | 2048 | imagenet_d | -38.23 | 2.96 | 41.19 |
| ema | 4 | 2048 | imagenet_v2 | -37.53 | 18.71 | 56.24 |
| tda | N.A. | N.A. | imagenet_v2 | -36.31 | 19.93 | 56.24 |

