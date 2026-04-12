"""Entrypoint canônico de carregamento de dados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_raw(
    path: str | Path,
    *,
    header: int = 1,
    id_col: str = "ID",
) -> pd.DataFrame:
    """Carrega o dataset bruto .xls, dropa a coluna ID e retorna DataFrame.

    Parâmetros
    ----------
    path   : caminho para o arquivo .xls original.
    header : linha usada como cabeçalho (padrão=1, pula a linha de título).
    id_col : nome da coluna identificadora a ser descartada.

    Returns
    -------
    pd.DataFrame com 30000 linhas e 24 colunas (sem coluna ID).

    Raises
    ------
    FileNotFoundError : se o arquivo não existir no path informado.
    ValueError        : se id_col não estiver presente no DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset bruto não encontrado: {path}")
    df = pd.read_excel(path, header=header)
    if id_col not in df.columns:
        raise ValueError(f"Coluna '{id_col}' não encontrada. Colunas: {list(df.columns)}")
    df = df.drop(columns=[id_col])
    return df.reset_index(drop=True)


def load_cleaned(path: str | Path) -> pd.DataFrame:
    """Carrega o parquet limpo produzido por build_clean_dataset.py.

    Raises
    ------
    FileNotFoundError : se o parquet não existir (rodar build_clean_dataset.py antes).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset limpo não encontrado: {path}\n"
            "Execute: python scripts/build_clean_dataset.py"
        )
    return pd.read_parquet(path, engine="pyarrow")


def load_config(config_path: str | Path | None = None) -> dict:
    """Carrega configs/data.yaml.

    Se config_path=None, resolve relativo à raiz do repo (dois níveis acima de src/).

    Returns
    -------
    dict com a estrutura completa do YAML.
    """
    if config_path is None:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        config_path = repo_root / "configs" / "data.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config não encontrado: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
