"""Convencao de nomes para runs MLflow."""

from __future__ import annotations


def compose_run_name(
    stage: str,
    model: str,
    preproc: str = "numstd_catoh",
    dimred: str = "none",
    search: str = "none",
    seed: int = 42,
    datahash8: str = "",
    githash7: str = "",
) -> str:
    """Gera nome padronizado para run MLflow.

    Formato:
    <stage>__<model>__<preproc>__<dimred>__<search>__seed<seed>__data<DATAHASH8>__code<GITHASH7>

    Parameters
    ----------
    stage     : 'baseline' ou 'tune'
    model     : 'perceptron', 'logreg', 'dtree', 'rf', 'gb'
    preproc   : descritor do preprocessamento (default: 'numstd_catoh')
    dimred    : descritor de reducao dimensional (default: 'none')
    search    : 'none', 'grid', 'random'
    seed      : semente usada
    datahash8 : primeiros 8 chars do SHA-256 do dataset
    githash7  : primeiros 7 chars do git commit hash

    Returns
    -------
    str no formato padronizado
    """
    return (
        f"{stage}__{model}__{preproc}__{dimred}__{search}"
        f"__seed{seed}__data{datahash8}__code{githash7}"
    )
