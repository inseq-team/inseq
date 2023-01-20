from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..utils import is_datasets_available
from .attribute import AttributeBaseArgs, attribute
from .base import BaseCLICommand

if is_datasets_available():
    from datasets import load_dataset


@dataclass
class AttributeDatasetArgs:
    dataset_name: str = field(
        metadata={
            "alias": "-d",
            "help": "The type of dataset to be loaded for attribution.",
        },
    )
    input_text_field: Optional[str] = field(
        metadata={"alias": "-f", "help": "Name of the field containing the input texts used for attribution."}
    )
    generated_text_field: Optional[str] = field(
        default=None,
        metadata={
            "alias": "-fgen",
            "help": "Name of the field containing the generated texts used for constrained decoding.",
        },
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"alias": "-dconf", "help": "The name of the Huggingface dataset configuration."}
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"alias": "-ddir", "help": "Path to the directory containing the data files."}
    )
    dataset_files: Optional[List[str]] = field(
        default=None, metadata={"alias": "-dfiles", "help": "Path to the dataset files."}
    )
    dataset_split: Optional[str] = field(default="train", metadata={"alias": "-dsplit", "help": "Dataset split."})
    dataset_revision: Optional[str] = field(
        default=None, metadata={"alias": "-drev", "help": "The Huggingface dataset revision."}
    )
    dataset_auth_token: Optional[str] = field(
        default=None, metadata={"alias": "-dauth", "help": "The auth token for the Huggingface dataset."}
    )


def load_fields_from_dataset(dataset_args: AttributeDatasetArgs) -> Tuple[List[str], Optional[List[str]]]:
    if not is_datasets_available():
        raise ImportError("The datasets library needs to be installed to use the attribute-dataset client.")
    dataset = load_dataset(
        dataset_args.dataset_name,
        dataset_args.dataset_config,
        data_dir=dataset_args.dataset_dir,
        data_files=dataset_args.dataset_files,
        split=dataset_args.dataset_split,
        revision=dataset_args.dataset_revision,
        use_auth_token=dataset_args.dataset_auth_token,
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
    _dataclasses = AttributeBaseArgs, AttributeDatasetArgs

    def run(args: Tuple[AttributeBaseArgs, AttributeDatasetArgs]):
        attribute_args, dataset_args = args
        input_texts, generated_texts = load_fields_from_dataset(dataset_args)
        attribute(input_texts, generated_texts, attribute_args)
