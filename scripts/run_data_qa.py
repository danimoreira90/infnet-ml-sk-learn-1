"""Orquestra fingerprint, schema, splits determinísticos e diagnósticos de qualidade."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permite rodar como script mesmo sem install -e
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from credit_default.data.diagnostics import run_all_diagnostics
from credit_default.data.fingerprint import compute_fingerprint, save_fingerprint
from credit_default.data.ingest import load_cleaned, load_config, load_raw
from credit_default.data.schema import save_schema, validate
from credit_default.data.splits import make_splits, save_split_indices, verify_splits


def main(config_path: str | None = None) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    qa_cfg = cfg["qa"]
    artifacts_cfg = cfg["artifacts"]
    reports_cfg = cfg["reports"]

    # ---------------------------------------------------------------------------
    # 1. Carregar dataset (parquet limpo; fallback para raw com aviso)
    # ---------------------------------------------------------------------------
    cleaned_path = repo_root / data_cfg["cleaned_path"]

    if cleaned_path.exists():
        print(f"[qa] Carregando parquet limpo: {cleaned_path}")
        df = load_cleaned(cleaned_path)
    else:
        raw_xls = repo_root.parent / data_cfg["raw_path"].replace("../", "")
        print(
            f"[qa] AVISO: parquet nao encontrado em {cleaned_path}. "
            f"Carregando .xls bruto: {raw_xls}. Execute build_clean_dataset.py primeiro.",
            file=sys.stderr,
        )
        df = load_raw(raw_xls, header=data_cfg["read_excel_header"], id_col=data_cfg["id_column"])

    print(f"[qa] Shape: {df.shape}")

    # ---------------------------------------------------------------------------
    # 2. Fingerprint
    # ---------------------------------------------------------------------------
    raw_xls_for_fp = repo_root.parent / data_cfg["raw_path"].replace("../", "")
    fp = compute_fingerprint(df, file_path=raw_xls_for_fp if raw_xls_for_fp.exists() else None)
    fp_path = repo_root / artifacts_cfg["fingerprint_path"]
    save_fingerprint(fp, fp_path)
    print(f"[qa] Fingerprint salvo: {fp_path}")
    print(f"[qa]   file_sha256 : {fp.get('file_sha256', 'N/A')[:16]}...")
    print(f"[qa]   file_short  : {fp.get('file_short', 'N/A')}")

    # ---------------------------------------------------------------------------
    # 3. Validar schema e salvar
    # ---------------------------------------------------------------------------
    print("[qa] Validando schema...")
    warnings = validate(
        df,
        expected_rows=None,  # contagem pode diferir apos dedup
        expected_cols=qa_cfg["expected_clean_cols"],
        target_col=data_cfg["target_column"],
        education_valid_codes=qa_cfg["education_valid_codes"],
        marriage_valid_codes=qa_cfg["marriage_valid_codes"],
    )
    schema_path = repo_root / artifacts_cfg["schema_path"]
    save_schema(df, schema_path, warnings=warnings)
    print(f"[qa] Schema salvo: {schema_path} ({len(warnings)} warning(s))")

    # ---------------------------------------------------------------------------
    # 4. Splits determinísticos
    # ---------------------------------------------------------------------------
    target_col = data_cfg["target_column"]
    print("[qa] Gerando splits 70/15/15 estratificados...")
    train_idx, val_idx, test_idx = make_splits(
        df,
        target_col,
        seed=split_cfg["seed"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )
    verify_splits(df, train_idx, val_idx, test_idx)
    print(f"[qa] Splits: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

    splits_path = repo_root / artifacts_cfg["split_indices_path"]
    save_split_indices(
        train_idx,
        val_idx,
        test_idx,
        seed=split_cfg["seed"],
        file_sha256=fp.get("file_sha256", ""),
        fingerprint_short=fp.get("file_short", ""),
        path=splits_path,
    )
    print(f"[qa] Split indices salvo: {splits_path}")

    # ---------------------------------------------------------------------------
    # 5. Diagnósticos (7 checks)
    # ---------------------------------------------------------------------------
    figures_dir = repo_root / reports_cfg["figures_dir"]
    print(f"[qa] Executando diagnosticos (figuras em: {figures_dir})...")
    diag = run_all_diagnostics(df, target_col, figures_dir=figures_dir, seed=split_cfg["seed"])

    # ---------------------------------------------------------------------------
    # 6. Salvar QA summary
    # ---------------------------------------------------------------------------
    qa_summary = {
        "fingerprint": fp,
        "schema_warnings": warnings,
        "splits": {
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "seed": split_cfg["seed"],
        },
        "diagnostics": diag,
    }
    qa_path = repo_root / artifacts_cfg["qa_summary_path"]
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"[qa] QA summary salvo: {qa_path}")

    # ---------------------------------------------------------------------------
    # 7. Resumo no stdout
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESUMO DA AUDITORIA DE DADOS")
    print("=" * 60)
    print(f"  Shape do dataset       : {df.shape}")
    print(f"  Missing total          : {diag['missing']['n_missing_total']}")
    print(f"  Duplicatas             : {diag['duplicates']['n_duplicate_rows']}")
    tgt = diag["target_distribution"]
    print(f"  Classe minoritaria     : {tgt['minority_class']} ({tgt['minority_ratio']:.1%})")
    print(f"  Imbalance warning      : {tgt['imbalance_warning']}")
    print(f"  Colunas c/ outliers    : {len(diag['outliers']['cols_with_outliers'])}")
    print(f"  Schema warnings        : {len(warnings)}")
    print(f"  Leakage risk           : {diag['leakage_risk']['severity']}")
    print(f"  file_short (DATAHASH8) : {fp.get('file_short', 'N/A')}")
    print("=" * 60)
    print("[qa] Concluido.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auditoria completa de dados para a Parte 2.")
    parser.add_argument("--config", default=None, help="Caminho para configs/data.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
