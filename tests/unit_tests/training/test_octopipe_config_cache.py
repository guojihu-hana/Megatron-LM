from types import SimpleNamespace

from megatron.training import global_vars


def teardown_function():
    global_vars.destroy_global_vars()


def test_get_octopipe_config_caches_yaml_config(monkeypatch):
    calls = []
    args = SimpleNamespace(octopipe=True, octopipe_config_yaml="config.yaml", octopipe_config_dir=None)
    global_vars.set_args(args)

    monkeypatch.setattr(
        global_vars,
        "_resolve_octopipe_config_path",
        lambda config_path, project_root: "/tmp/config.yaml",
    )

    import octopipe.generate_inst as generate_inst

    def fake_get_octopipe_config_from_yaml(yaml_path):
        calls.append(yaml_path)
        return {"stage_num": 16}

    monkeypatch.setattr(
        generate_inst, "get_octopipe_config_from_yaml", fake_get_octopipe_config_from_yaml
    )

    assert global_vars.get_octopipe_config() == {"stage_num": 16}
    assert global_vars.get_octopipe_config() == {"stage_num": 16}
    assert calls == ["/tmp/config.yaml"]
