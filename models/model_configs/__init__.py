import json
from pathlib import Path


_CONFIG_DIR = Path(__file__).resolve().parent


def list_available_models() -> list[str]:
    """Return all available model config names."""
    model_names = []
    for config_path in _CONFIG_DIR.glob("*.json"):
        if config_path.name == "schema.json":
            continue
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        model_names.append(config["model_name"])
    return sorted(model_names)


def load_model_config(model_name: str) -> dict:
    """Load a model config by model_name."""
    config_path = _CONFIG_DIR / f"{model_name}.json"
    if not config_path.exists():
        available = ", ".join(list_available_models())
        raise ValueError(
            f"Unknown model_name '{model_name}'. Available models: {available}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if config.get("model_name") != model_name:
        raise ValueError(
            f"Config file {config_path} declares model_name "
            f"'{config.get('model_name')}', expected '{model_name}'"
        )
    return config
