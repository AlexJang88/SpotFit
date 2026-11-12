from app.services import property_model_bootstrap as boot


def _make_outputs(root, specs=boot.SPECS):
    output = root / "data" / "processed" / "output"
    output.mkdir(parents=True)
    for spec in specs:
        for source, _ in spec.outputs:
            (output / source).write_bytes(b"model")


def test_existing_models_are_left_alone(tmp_path):
    _make_outputs(tmp_path)
    assert set(boot.ensure_models(tmp_path).values()) == {"copied"}
    assert set(boot.ensure_models(tmp_path).values()) == {"exists"}
    assert all(boot.is_ready(spec, tmp_path) for spec in boot.SPECS)


def test_missing_models_are_trained_then_copied(tmp_path, monkeypatch):
    calls = []

    def fake_train(spec, root):
        calls.append(spec.name)
        output = root / "data" / "processed" / "output"
        output.mkdir(parents=True, exist_ok=True)
        for source, _ in spec.outputs:
            (output / source).write_bytes(b"trained")
        return 0

    monkeypatch.setattr(boot, "_train", fake_train)
    assert set(boot.ensure_models(tmp_path).values()) == {"trained"}
    assert calls == [spec.name for spec in boot.SPECS]
    model = tmp_path / "models" / "recommender" / "v1" / "apart_wolse_final" / "model_bundle.joblib"
    assert model.read_bytes() == b"trained"


def test_one_failure_does_not_stop_the_others(tmp_path, monkeypatch):
    _make_outputs(tmp_path, boot.SPECS[1:])          # sale outputs are missing and training fails
    monkeypatch.setattr(boot, "_train", lambda spec, root: 1)
    result = boot.ensure_models(tmp_path)
    assert result["apart_sale_final"].startswith("failed")
    assert result["apart_jeonse_final"] == "copied" and result["apart_wolse_final"] == "copied"
    assert not boot.is_ready(boot.SPECS[0], tmp_path)


def test_missing_data_folder_is_reported_not_raised(tmp_path):
    result = boot.ensure_models(tmp_path)             # no scripts, no data
    assert all(value.startswith("failed") for value in result.values())


def test_bootstrap_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPERTY_MODEL_BOOTSTRAP", "0")
    assert boot.start_background_bootstrap(tmp_path) is False
