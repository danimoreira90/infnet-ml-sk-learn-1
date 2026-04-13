# Comparacao de Modelos — Tune

> Ordenado por roc_auc descendente (metrica primaria).
> Gerado automaticamente por `scripts/generate_comparison_table.py`.

| run_name | model_family | roc_auc | f1_macro | precision_macro | recall_macro | accuracy | cv_roc_auc_mean | cv_roc_auc_std | cv_f1_mean | cv_f1_std | training_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tune__gb__numstd_catoh__none__random__seed42__data30c6be3a__codebdd19bf | ensemble | 0.7794 | 0.6792 | 0.7506 | 0.6548 | 0.8180 | 0.7830 | 0.0048 | 0.6823 | 0.0057 | 404.5295 |
| tune__rf__numstd_catoh__none__random__seed42__data30c6be3a__codebdd19bf | ensemble | 0.7767 | 0.6691 | 0.7468 | 0.6454 | 0.8151 | 0.7669 | 0.0062 | 0.6793 | 0.0057 | 74.8223 |
| tune__dtree__numstd_catoh__none__grid__seed42__data30c6be3a__codebdd19bf | tree | 0.7514 | 0.6771 | 0.7429 | 0.6538 | 0.8154 | 0.6209 | 0.0047 | 0.6178 | 0.0055 | 7.6187 |
| tune__logreg__numstd_catoh__none__grid__seed42__data30c6be3a__codebdd19bf | linear | 0.7234 | 0.6111 | 0.7544 | 0.5968 | 0.8062 | 0.7273 | 0.0028 | 0.6238 | 0.0040 | 12.5763 |
| tune__perceptron__numstd_catoh__none__grid__seed42__data30c6be3a__codebdd19bf | linear | 0.6912 | 0.4410 | 0.8897 | 0.5015 | 0.7795 | 0.6998 | 0.0096 | 0.4429 | 0.0044 | 3.1524 |
