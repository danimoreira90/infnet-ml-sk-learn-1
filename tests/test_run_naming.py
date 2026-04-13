"""Testes para convencao de nomes de runs MLflow."""

from credit_default.tracking.run_naming import compose_run_name


def test_compose_run_name_baseline_format():
    name = compose_run_name("baseline", "rf", datahash8="30c6be3a", githash7="1c9dc04")
    assert name == "baseline__rf__numstd_catoh__none__none__seed42__data30c6be3a__code1c9dc04"


def test_compose_run_name_tune_format():
    name = compose_run_name(
        "tune", "rf", search="random", datahash8="30c6be3a", githash7="1c9dc04"
    )
    assert name == "tune__rf__numstd_catoh__none__random__seed42__data30c6be3a__code1c9dc04"


def test_compose_run_name_custom_params():
    name = compose_run_name(
        "baseline",
        "logreg",
        preproc="custom",
        dimred="pca",
        seed=123,
        datahash8="abcd1234",
        githash7="abc1234",
    )
    assert name == "baseline__logreg__custom__pca__none__seed123__dataabcd1234__codeabc1234"


def test_compose_run_name_defaults():
    name = compose_run_name("baseline", "dtree")
    assert "__numstd_catoh__" in name
    assert "__none__none__" in name
    assert "__seed42__" in name


def test_compose_run_name_all_models():
    for model in ["perceptron", "logreg", "dtree", "rf", "gb"]:
        name = compose_run_name("baseline", model)
        assert f"__{model}__" in name
