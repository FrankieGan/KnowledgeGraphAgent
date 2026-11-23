"""Read configuration from the environment and an untracked local JSON file."""

import json
import os
from pathlib import Path


def _get_local_configuration_variable(name: str) -> str | None:
    config_path = Path(__file__).with_name("config.local.json")
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8-sig") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config.local.json must contain a JSON object")
    value = config.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def get_environment_variable(name: str) -> str | None:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    if os.name != "nt":
        return _get_local_configuration_variable(name)

    import winreg

    # Running IDEs may predate a variable added through Windows Settings.
    for hive, path in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ):
        try:
            with winreg.OpenKey(hive, path) as key:
                value, kind = winreg.QueryValueEx(key, name)
            if kind in (winreg.REG_SZ, winreg.REG_EXPAND_SZ) and isinstance(value, str) and value.strip():
                return value.strip()
        except OSError:
            continue
    return _get_local_configuration_variable(name)
