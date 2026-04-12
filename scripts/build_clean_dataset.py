"""Reconstrói o dataset limpo (parquet) a partir do .xls bruto verificado."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite rodar como script mesmo sem install -e
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from credit_default.data.fingerprint import compute_file_sha256
from credit_default.data.ingest import load_config, load_raw
from credit_default.data.schema import DataValidationError, validate


def main(config_path: str | None = None, raw_path_override: str | None = None) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    qa_cfg = cfg["qa"]

    # Resolve raw_path relativo à raiz do repo (dois níveis acima de scripts/)
    repo_root = Path(__file__).resolve().parent.parent
    raw_path = Path(raw_path_override) if raw_path_override else (repo_root / data_cfg["raw_path"])

    print(f"[build] Dataset bruto: {raw_path}")
    if not raw_path.exists():
        print(f"ERRO: arquivo não encontrado: {raw_path}", file=sys.stderr)
        sys.exit(1)

    # Verificar SHA-256
    print("[build] Calculando SHA-256 do arquivo bruto...")
    actual_sha = compute_file_sha256(raw_path)
    expected_sha = data_cfg["expected_sha256"]
    if actual_sha != expected_sha:
        print(
            f"ERRO: SHA-256 diverge!\n  Esperado : {expected_sha}\n  Encontrado: {actual_sha}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[build] SHA-256 verificado: {actual_sha[:8]}…")

    # Carregar
    print("[build] Carregando dataset...")
    df = load_raw(raw_path, header=data_cfg["read_excel_header"], id_col=data_cfg["id_column"])
    print(f"[build] Shape após ingestão: {df.shape}")

    # Remover duplicatas exatas antes da validação (achadas no raw; registradas explicitamente)
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(
            f"[build] AVISO: {n_dropped} linha(s) exatamente duplicada(s) removida(s) "
            f"({n_before} -> {len(df)} linhas). Fato registrado; dataset ainda valido."
        )

    # Validar (expected_rows=None pois contagem pode diferir após dedup)
    print("[build] Validando schema...")
    try:
        warnings = validate(
            df,
            expected_rows=None,
            expected_cols=qa_cfg["expected_clean_cols"],
            target_col=data_cfg["target_column"],
            education_valid_codes=qa_cfg["education_valid_codes"],
            marriage_valid_codes=qa_cfg["marriage_valid_codes"],
        )
    except DataValidationError as exc:
        print(f"ERRO de validação: {exc}", file=sys.stderr)
        sys.exit(1)

    for w in warnings:
        print(f"[build] AVISO: {w}")

    # Salvar parquet
    out_path = repo_root / data_cfg["cleaned_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"[build] Parquet salvo em: {out_path}")
    print(f"[build] Shape final: {df.shape} | Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
    print("[build] Concluído.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstrói o dataset limpo a partir do .xls bruto."
    )
    parser.add_argument("--config", default=None, help="Caminho para configs/data.yaml")
    parser.add_argument(
        "--raw-path", default=None, dest="raw_path", help="Override do raw_path do config"
    )
    args = parser.parse_args()
    main(config_path=args.config, raw_path_override=args.raw_path)
