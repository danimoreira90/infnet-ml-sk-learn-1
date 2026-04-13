"""Testes para log_standard_params em mlflow_utils."""

from __future__ import annotations

import mlflow
import pytest

from credit_default.tracking.mlflow_utils import log_standard_params


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    """Garante tracking URI isolado por teste."""
    mlflow.set_tracking_uri(tmp_path.as_uri())
    yield
    mlflow.set_tracking_uri("")


def _run_with_params(
    clf_params: dict,
    search_type: str = "none",
    dimred_method: str = "none",
    dimred_n_components: int = 0,
) -> dict:
    import uuid

    exp_id = mlflow.create_experiment(f"test-{uuid.uuid4().hex[:8]}")
    with mlflow.start_run(experiment_id=exp_id) as run:
        log_standard_params(
            run,
            model_name="logreg",
            seed=42,
            cv_folds=5,
            n_train=20975,
            n_val=4495,
            search_type=search_type,
            clf_params=clf_params,
            dimred_method=dimred_method,
            dimred_n_components=dimred_n_components,
        )
        run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    return client.get_run(run_id).data.params


def test_log_params_contains_meta_keys():
    """Os 5 params meta obrigatorios estao presentes."""
    params = _run_with_params({"clf__C": 1.0, "clf__solver": "lbfgs", "clf__max_iter": "500"})
    assert params["model_name"] == "logreg"
    assert params["seed"] == "42"
    assert params["cv_folds"] == "5"
    assert params["scoring_primary"] == "roc_auc"
    assert params["split_strategy"] == "stratified_70_15_15_from_part2"


def test_log_params_contains_context_keys():
    """search_type, n_train, n_val presentes."""
    params = _run_with_params(
        {"clf__C": 1.0, "clf__solver": "lbfgs", "clf__max_iter": "500"}, search_type="grid"
    )
    assert params["search_type"] == "grid"
    assert params["n_train"] == "20975"
    assert params["n_val"] == "4495"


def test_log_params_clf_prefix_keys_present():
    """Params clf__* do classificador estao loggeados."""
    clf_params = {"clf__C": 0.1, "clf__solver": "saga", "clf__max_iter": 500}
    params = _run_with_params(clf_params)
    clf_keys = [k for k in params if k.startswith("clf__")]
    assert len(clf_keys) >= 3, f"Esperado >= 3 clf__ params, encontrado {clf_keys}"


def test_log_params_minimum_total_count():
    """DoD: cada run tem minimo 8 params (5 meta + search_type + n_train + n_val)
    mais ao menos 3 clf__ hyperparams = 11 total minimo."""
    clf_params = {
        "clf__max_depth": 5,
        "clf__criterion": "gini",
        "clf__min_samples_split": 2,
    }
    params = _run_with_params(clf_params, search_type="grid")
    assert len(params) >= 8, (
        f"Esperado >= 8 params totais, encontrado {len(params)}: {list(params.keys())}"
    )
    clf_keys = [k for k in params if k.startswith("clf__")]
    assert len(clf_keys) >= 3, f"Esperado >= 3 clf__ params, encontrado {len(clf_keys)}"


def test_log_params_tuned_best_params():
    """Para runs tuned, best_params_ aparecem com prefixo clf__."""
    best_params = {
        "clf__n_estimators": 300,
        "clf__max_depth": 10,
        "clf__min_samples_split": 2,
        "clf__max_features": "sqrt",
    }
    params = _run_with_params(best_params, search_type="random")
    assert params["clf__n_estimators"] == "300"
    assert params["clf__max_depth"] == "10"
    assert params["clf__max_features"] == "sqrt"


def test_log_params_dimred_pca_keys_present():
    """Params dimred PCA: dimred_method e dimred_n_components presentes."""
    params = _run_with_params(
        {"clf__C": 1.0},
        search_type="none",
        dimred_method="pca",
        dimred_n_components=10,
    )
    assert params["dimred_method"] == "pca"
    assert params["dimred_n_components"] == "10"
    assert params["scoring_primary"] == "roc_auc"


def test_log_params_dimred_lda_keys_present():
    """Params dimred LDA: dimred_method=lda e dimred_n_components=1."""
    params = _run_with_params(
        {"clf__C": 1.0},
        search_type="none",
        dimred_method="lda",
        dimred_n_components=1,
    )
    assert params["dimred_method"] == "lda"
    assert params["dimred_n_components"] == "1"
