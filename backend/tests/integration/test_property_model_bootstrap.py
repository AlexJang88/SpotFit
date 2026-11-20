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


def _forecast_csv(path, current, forecast=(105.0, 110.0, 115.0), rows=20):
    import pandas as pd
    frame = pd.DataFrame({"행정동_코드": range(rows), "자치구": "a", "행정동": "b",
                          "현재 평당 금액(최근12개월 실거래, 만원)": current,
                          "1년 뒤 예측(만원/평)": forecast[0], "2년 뒤 예측(만원/평)": forecast[1],
                          "3년 뒤 예측(만원/평)": forecast[2]})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def test_refresh_rejects_a_broken_forecast_and_keeps_the_served_model(tmp_path, monkeypatch):
    spec = boot.SPECS[0]
    served = tmp_path / "models" / "recommender" / "v1" / spec.name
    _forecast_csv(served / "neighborhood_forecasts.csv", current=100.0)
    output = tmp_path / "data" / "processed" / "output"
    csv_name = next(src for src, dest in spec.outputs if dest == "neighborhood_forecasts.csv")
    _forecast_csv(output / csv_name, current=100.0, rows=10)                 # lost half of the dongs
    assert any("동네 수" in problem for problem in boot.validate_candidate(spec, tmp_path))
    _forecast_csv(output / csv_name, current=300.0)                          # prices tripled
    assert any("달라졌습니다" in problem for problem in boot.validate_candidate(spec, tmp_path))
    _forecast_csv(output / csv_name, current=103.0, forecast=(0.0, 110.0, 115.0))
    assert any("0 이하" in problem for problem in boot.validate_candidate(spec, tmp_path))
    _forecast_csv(output / csv_name, current=103.0)
    assert boot.validate_candidate(spec, tmp_path) == []


def test_refresh_replaces_models_only_after_validation(tmp_path, monkeypatch):
    for directory in ("ml/training", "ml/etl", "data/processed"):
        (tmp_path / directory).mkdir(parents=True)
    for spec in boot.SPECS:
        (tmp_path / "ml" / "training" / spec.script).write_text("")
    seen = {}

    def fake_run(script, cwd, log_path, label, extra_env=None):
        seen[script.name] = extra_env
        spec = next(s for s in boot.SPECS if s.script == script.name)
        output = tmp_path / "data" / "processed" / "output"
        output.mkdir(parents=True, exist_ok=True)
        for source, dest in spec.outputs:
            if dest == "neighborhood_forecasts.csv":
                _forecast_csv(output / source, current=100.0)
            else:
                (output / source).write_bytes(b"new")
        return 0

    monkeypatch.setattr(boot, "_run_script", fake_run)
    monkeypatch.setattr(boot, "_build_input", lambda spec, root, force=False: None)
    monkeypatch.setattr(boot, "validate_candidate", lambda spec, root=None: ["번들 검증 실패"] if spec.kind == "bundle" else [])
    result = boot.refresh_models(tmp_path, data_end="2026-10-31", base_month="2026-10")
    assert result["apart_sale_final"] == result["apart_jeonse_final"] == "refreshed"
    assert result["apart_wolse_final"].startswith("rejected")
    assert all(env == {"PROPERTY_DATA_END": "2026-10-31"} for env in seen.values())
    root = tmp_path / "models" / "recommender" / "v1"
    import json
    meta = json.loads((root / "apart_sale_final" / boot.META_NAME).read_text(encoding="utf-8"))
    assert meta["base_year_month"] == "2026-10"
    assert not (root / "apart_wolse_final").exists()


def test_missing_input_file_is_built_before_training(tmp_path, monkeypatch):
    (tmp_path / "ml" / "training").mkdir(parents=True)
    (tmp_path / "ml" / "etl").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    spec = boot.SPECS[0]
    (tmp_path / "ml" / "training" / spec.script).write_text("")
    (tmp_path / "ml" / "etl" / spec.build).write_text("")
    ran = []

    def fake_run(script, cwd, log_path, label, extra_env=None):
        ran.append(script.name)
        if script.name == spec.build:
            target = tmp_path / "data" / "processed" / spec.input_file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x")
        return 0

    monkeypatch.setattr(boot, "_run_script", fake_run)
    assert boot._train(spec, tmp_path) == 0
    assert ran == [spec.build, spec.script]
    ran.clear()
    assert boot._train(spec, tmp_path) == 0           # input file already exists -> only training runs
    assert ran == [spec.script]
