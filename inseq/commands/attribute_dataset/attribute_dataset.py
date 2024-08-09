from ...utils import is_datasets_available
from ..attribute import AttributeExtendedArgs
from ..attribute.attribute import attribute
from ..base import BaseCLICommand
from .attribute_dataset_args import LoadDatasetArgs

if is_datasets_available():
    from datasets import load_dataset


def load_fields_from_dataset(dataset_args: LoadDatasetArgs) -> tuple[list[str], list[str] | None]:
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

    def run(args: tuple[AttributeExtendedArgs, LoadDatasetArgs]):
        attribute_args, dataset_args = args
        input_texts, generated_texts = load_fields_from_dataset(dataset_args)
        attribute(input_texts, generated_texts, attribute_args)
