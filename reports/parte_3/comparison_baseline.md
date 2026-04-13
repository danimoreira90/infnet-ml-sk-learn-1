# Comparacao de Modelos — Baseline

> Ordenado por roc_auc descendente (metrica primaria).
> Gerado automaticamente por `scripts/generate_comparison_table.py`.

| run_name | model_family | roc_auc | f1_macro | precision_macro | recall_macro | accuracy | cv_roc_auc_mean | cv_roc_auc_std | cv_f1_mean | cv_f1_std | training_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline__gb__numstd_catoh__none__none__seed42__data30c6be3a__code0284dc2 | ensemble | 0.7795 | 0.6733 | 0.7505 | 0.6491 | 0.8169 | 0.7830 | 0.0048 | 0.6823 | 0.0057 | 6.9776 |
| baseline__rf__numstd_catoh__none__none__seed42__data30c6be3a__code0284dc2 | ensemble | 0.7573 | 0.6783 | 0.7459 | 0.6545 | 0.8165 | 0.7669 | 0.0062 | 0.6793 | 0.0057 | 4.1960 |
| baseline__logreg__numstd_catoh__none__none__seed42__data30c6be3a__code0284dc2 | linear | 0.7232 | 0.6111 | 0.7544 | 0.5968 | 0.8062 | 0.7273 | 0.0028 | 0.6238 | 0.0040 | 0.1493 |
| baseline__perceptron__numstd_catoh__none__none__seed42__data30c6be3a__code0284dc2 | linear | 0.6931 | 0.4459 | 0.6977 | 0.5033 | 0.7795 | 0.6998 | 0.0096 | 0.4429 | 0.0044 | 0.0964 |
| baseline__dtree__numstd_catoh__none__none__seed42__data30c6be3a__code0284dc2 | tree | 0.5987 | 0.5982 | 0.5979 | 0.5984 | 0.7224 | 0.6209 | 0.0047 | 0.6178 | 0.0055 | 0.4372 |
