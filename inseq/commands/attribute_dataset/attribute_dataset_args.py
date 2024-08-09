from dataclasses import dataclass

from ...utils import cli_arg
from ..commands_utils import command_args_docstring


@command_args_docstring
@dataclass
class LoadDatasetArgs:
    dataset_name: str = cli_arg(
        aliases=["-d", "--dataset"],
        help="The type of dataset to be loaded for attribution.",
    )
    input_text_field: str | None = cli_arg(
        aliases=["-in", "--input"], help="Name of the field containing the input texts used for attribution."
    )
    generated_text_field: str | None = cli_arg(
        default=None,
        aliases=["-gen", "--generated"],
        help="Name of the field containing the generated texts used for constrained decoding.",
    )
    dataset_config: str | None = cli_arg(
        default=None, aliases=["--config"], help="The name of the Huggingface dataset configuration."
    )
    dataset_dir: str | None = cli_arg(
        default=None, aliases=["--dir"], help="Path to the directory containing the data files."
    )
    dataset_files: list[str] | None = cli_arg(default=None, aliases=["--files"], help="Path to the dataset files.")
    dataset_split: str | None = cli_arg(default="train", aliases=["--split"], help="Dataset split.")
    dataset_revision: str | None = cli_arg(
        default=None, aliases=["--revision"], help="The Huggingface dataset revision."
    )
    dataset_auth_token: str | None = cli_arg(
        default=None, aliases=["--auth"], help="The auth token for the Huggingface dataset."
    )
    dataset_kwargs: dict | None = cli_arg(
        default_factory=dict,
        help="Additional keyword arguments passed to the dataset constructor in JSON format.",
    )
