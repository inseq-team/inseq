from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..utils import cli_arg, is_datasets_available
from .attribute import AttributeExtendedArgs, attribute
from .base import BaseCLICommand

if is_datasets_available():
    from datasets import load_dataset


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


def load_fields_from_dataset(dataset_args: LoadDatasetArgs) -> Tuple[List[str], Optional[List[str]]]:
    if not is_datasets_available():
        raise ImportError("The datasets library needs to be installed to use the attribute-dataset client.")
    dataset = load_dataset(
        dataset_args.dataset_name,
        dataset_args.dataset_config,
        data_dir=dataset_args.dataset_dir,
        data_files=dataset_args.dataset_files,
        split=dataset_args.dataset_split,
        revision=dataset_args.dataset_revision,
        token=dataset_args.dataset_auth_token,
        **dataset_args.dataset_kwargs,
    )
    df = dataset.to_pandas()
    if dataset_args.input_text_field in df.columns:
        input_texts = list(df[dataset_args.input_text_field])
    else:
        raise ValueError(f"The input text field {dataset_args.input_text_field} does not exist in the dataset.")
    generated_texts = None
    if dataset_args.generated_text_field is not None:
        if dataset_args.generated_text_field in df.columns:
            generated_texts = list(df[dataset_args.generated_text_field])
    return input_texts, generated_texts


class AttributeDatasetCommand(BaseCLICommand):
    _name = "attribute-dataset"
    _help = "Perform feature attribution on a full dataset and save the results to a file"
    _dataclasses = AttributeExtendedArgs, LoadDatasetArgs

    def run(args: Tuple[AttributeExtendedArgs, LoadDatasetArgs]):
        attribute_args, dataset_args = args
        input_texts, generated_texts = load_fields_from_dataset(dataset_args)
        attribute(input_texts, generated_texts, attribute_args)
