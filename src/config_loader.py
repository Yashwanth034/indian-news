"""Central config loading + JSON-schema validation for IndiaNews.

Every config file in config/ is validated against its schema in
config/schemas/ at load time, so a typo or wrong type fails fast at
startup (and in CI) rather than at runtime.
"""
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
SCHEMA_DIR = CONFIG_DIR / "schemas"

# Mapping of config file name -> schema file name.
CONFIG_FILES = {
    "config.json": "config.schema.json",
    "sources.json": "sources.schema.json",
    "categories.json": "categories.schema.json",
    "editorial.json": "editorial.schema.json",
    "india_entities.json": "india_entities.schema.json",
    "india_geo.json": "india_geo.schema.json",
}


class ConfigError(Exception):
    """Raised when a config file is missing or fails schema validation."""


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _validate(data, schema, label):
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    if errors:
        details = "\n".join(
            f"  - {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        )
        raise ConfigError(f"{label} failed schema validation:\n{details}")


def _load_config():
    config = {}
    for file_name, schema_name in CONFIG_FILES.items():
        path = CONFIG_DIR / file_name
        schema_path = SCHEMA_DIR / schema_name
        if not path.exists():
            raise ConfigError(f"missing config file: {path}")
        if not schema_path.exists():
            raise ConfigError(f"missing schema file: {schema_path}")
        data = _load_json(path)
        schema = _load_json(schema_path)
        _validate(data, schema, file_name)
        config[file_name.replace(".json", "")] = data
    return config


_config = None


def get_config():
    """Load (once) and return the validated configuration bundle."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def reload():
    """Force a reload on next get_config() call. Used by tests."""
    global _config
    _config = None
    return get_config()
