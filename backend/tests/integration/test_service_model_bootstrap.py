from app.services import service_model_bootstrap as boot


def create_required(root, group):
    base = root / "models/recommender/v1"
    for relative in boot.GROUPS[group]["required"]:
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")


def test_ready_groups_are_not_rebuilt(tmp_path, monkeypatch):
    for group in boot.GROUPS:
        create_required(tmp_path, group)
    monkeypatch.setattr(boot, "_run_script", lambda *_: (_ for _ in ()).throw(AssertionError()))
    assert set(boot.ensure_all_models(tmp_path).values()) == {"exists"}
    assert boot.bootstrap_status(tmp_path)["ready"] is True


def test_missing_groups_are_rebuilt_and_rechecked(tmp_path, monkeypatch):
    calls = []
    def rebuild(group, root):
        calls.append(group)
        create_required(root, group)
        return 0
    monkeypatch.setattr(boot, "_run_script", rebuild)
    assert set(boot.ensure_all_models(tmp_path).values()) == {"restored"}
    assert calls == list(boot.GROUPS)
    assert boot.bootstrap_status(tmp_path)["ready"] is True


def test_exit_zero_is_not_success_when_artifacts_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "_run_script", lambda *_: 0)
    result = boot.ensure_all_models(tmp_path)
    assert all(value.startswith("failed") for value in result.values())
    assert boot.bootstrap_status(tmp_path)["ready"] is False


def test_one_failure_does_not_stop_later_groups(tmp_path, monkeypatch):
    def rebuild(group, root):
        if group != "property":
            create_required(root, group)
        return 1 if group == "property" else 0
    monkeypatch.setattr(boot, "_run_script", rebuild)
    result = boot.ensure_all_models(tmp_path)
    assert result["property"].startswith("failed")
    assert result["traffic"] == result["child_crime"] == "restored"
