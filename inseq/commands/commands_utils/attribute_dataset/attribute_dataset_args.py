from dataclasses import dataclass
from typing import List, Optional

from ....utils import cli_arg


@dataclass
class LoadDatasetArgs:
    dataset_name: str = cli_arg(
        aliases=["-d", "--dataset"],
        help="The type of dataset to be loaded for attribution.",
    )
    input_text_field: Optional[str] = cli_arg(
        aliases=["-in", "--input"], help="Name of the field containing the input texts used for attribution."
    )
    generated_text_field: Optional[str] = cli_arg(
        default=None,
        aliases=["-gen", "--generated"],
        help="Name of the field containing the generated texts used for constrained decoding.",
    )
    dataset_config: Optional[str] = cli_arg(
        default=None, aliases=["--config"], help="The name of the Huggingface dataset configuration."
    )
    dataset_dir: Optional[str] = cli_arg(
        default=None, aliases=["--dir"], help="Path to the directory containing the data files."
    )
    dataset_files: Optional[List[str]] = cli_arg(default=None, aliases=["--files"], help="Path to the dataset files.")
    dataset_split: Optional[str] = cli_arg(default="train", aliases=["--split"], help="Dataset split.")
    dataset_revision: Optional[str] = cli_arg(
        default=None, aliases=["--revision"], help="The Huggingface dataset revision."
    )
    dataset_auth_token: Optional[str] = cli_arg(
        default=None, aliases=["--auth"], help="The auth token for the Huggingface dataset."
    )
    dataset_kwargs: Optional[dict] = cli_arg(
        default_factory=dict,
        help="Additional keyword arguments passed to the dataset constructor in JSON format.",
    )
