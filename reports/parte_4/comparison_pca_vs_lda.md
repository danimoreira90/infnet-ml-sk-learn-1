# Pivot: ROC-AUC por Modelo e Tecnica de Dimred

Colunas: baseline sem dimred (P3) + 3 configs Parte 4.
Metrica primaria: `roc_auc`.

| model      |   none (P3) |   pca_k10 |   pca_k15 |   lda_k1 |
|:-----------|------------:|----------:|----------:|---------:|
| perceptron |      0.6931 |    0.6591 |    0.67   |   0.7171 |
| logreg     |      0.7232 |    0.7045 |    0.7167 |   0.7171 |
| dtree      |      0.5987 |    0.6006 |    0.6096 |   0.5968 |
| rf         |      0.7573 |    0.742  |    0.7417 |   0.653  |
| gb         |      0.7795 |    0.762  |    0.7661 |   0.7189 |
