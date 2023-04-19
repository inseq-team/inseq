import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    attention_module: str
    value_vector: str


MODEL_CONFIGS = {
    model_type: ModelConfig(**cfg)
    for model_type, cfg in yaml.safe_load(open(Path(__file__).parent / "model_config.yaml", encoding="utf8")).items()
}


def get_model_config(model_type: str) -> ModelConfig:
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"A configuration for the {model_type} model is not defined. "
            "You can register a configuration with :meth:`~inseq.models.register_model_config`, "
            "or request it to be added to the library by opening an issue on GitHub: "
            "https://github.com/inseq-team/inseq/issues"
        )
    return MODEL_CONFIGS[model_type]


def register_model_config(
    model_type: str,
    config: dict,
    overwrite: bool = False,
    allow_partial: bool = False,
) -> None:
    if model_type in MODEL_CONFIGS:
        if not overwrite:
            raise ValueError(
                f"{model_type} is already registered in model configurations.Override with overwrite=True."
            )
        logger.warning(f"Overwriting {model_type} config.")
    all_fields = set(ModelConfig.__dataclass_fields__.keys())
    config_fields = set(config.keys())
    diff = all_fields - config_fields
    if diff and not allow_partial:
        raise ValueError(
            f"Missing fields {','.join(diff)} in model configuration for {model_type}."
            "Set allow_partial=True to allow partial configuration."
        )
    if allow_partial:
        config = {**{field: None for field in diff}, **config}
    MODEL_CONFIGS[model_type] = config
