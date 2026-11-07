import pytest
from pydantic import ValidationError

from quantlab.settings import CONFIG_FILES, load_config, load_settings, parse_override


def test_all_shipped_configs_validate(config_dir):
    for name in CONFIG_FILES:
        load_config(name, config_dir)


def test_env_override_beats_yaml(config_dir):
    s = load_config("settings", config_dir, environ={"QUANTLAB__DATA__FEED": "sip"})
    assert s.data.feed == "sip"


def test_env_override_can_target_other_files(config_dir):
    c = load_config("impact", config_dir, environ={"QUANTLAB__IMPACT__SIMULATION__SEED": "123"})
    assert c.simulation.seed == 123


def test_explicit_override_beats_env(config_dir):
    s = load_config("settings", config_dir, overrides={"data.feed": "delayed_sip"}, environ={"QUANTLAB__DATA__FEED": "sip"})
    assert s.data.feed == "delayed_sip"


def test_parse_override_keeps_types():
    assert parse_override("vol.windows=[10, 30]") == ("vol.windows", [10, 30])
    assert parse_override("trading.allow_live=false") == ("trading.allow_live", False)


def test_unknown_keys_rejected(config_dir):
    with pytest.raises(ValidationError):
        load_settings(config_dir, overrides={"data.typo_key": 1})


def test_window_longer_than_history_rejected(config_dir):
    with pytest.raises(ValidationError):
        load_settings(config_dir, overrides={"data.lookback_days": 30})


def test_param_spec_default_must_be_in_bounds(config_dir):
    with pytest.raises(ValidationError):
        load_config("impact", config_dir, overrides={"params.eta.default": 99.0})
