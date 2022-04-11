from typing import List, Optional, Tuple

from dataclasses import dataclass, field

from .. import load_model
from ..utils import is_datasets_available
from .attribute import AttributeArgs
from .base import BaseCLICommand


if is_datasets_available():
    from datasets import load_dataset


@dataclass
class AttributeDatasetArgs:
    dataset_name: str = field(
        metadata={
            "help": "The type of dataset to be loaded for attribution.",
        },
    )
    input_text_field: Optional[str] = field(
        metadata={"help": "Name of the field containing the input texts used for attribution."}
    )
    generated_texts_field: Optional[str] = field(
        metadata={"help": "Name of the field containing the generated texts used for constrained decoding."}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "The name of the Huggingface dataset configuration."}
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to the directory containing the data files."}
    )
    dataset_files: Optional[List[str]] = field(default=None, metadata={"help": "Path to the dataset files."})
    dataset_split: Optional[str] = field(default="train", metadata={"help": "Dataset split."})
    dataset_revision: Optional[str] = field(default=None, metadata={"help": "The Huggingface dataset revision."})
    dataset_auth_token: Optional[str] = field(
        default=None, metadata={"help": "The auth token for the Huggingface dataset."}
    )
    batch_size: Optional[int] = field(
        default=8, metadata={"help": "The batch size to use for iterating over the dataset"}
    )


class AttributeDatasetCommand(BaseCLICommand):
    _name = "attribute-dataset"
    _help = "Perform feature attribution on a full dataset and save the results to a file"
    _dataclasses = AttributeArgs, AttributeDatasetArgs

    def run(args: Tuple[AttributeArgs, AttributeDatasetArgs]):
        attribute_args, dataset_args = args
        if not is_datasets_available():
            raise ImportError("Datasets are not available. Please install Huggingface's datasets library.")
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
        if dataset_args.generated_texts_field is not None:
            if dataset_args.generated_texts_field in df.columns:
                generated_texts = list(df[dataset_args.generated_texts_field])
        model = load_model(attribute_args.model_name_or_path, attribution_method=attribute_args.attribution_method)
        out = model.attribute(
            input_texts,
            generated_texts,
            attribute_target=attribute_args.do_prefix_attribution,
            output_step_probabilities=attribute_args.output_step_probabilities,
            output_step_attributions=attribute_args.output_step_attributions,
            include_eos_baseline=attribute_args.include_eos_baseline,
            n_steps=attribute_args.n_approximation_steps,
            internal_batch_size=attribute_args.attribution_batch_size,
            return_convergence_delta=attribute_args.return_convergence_delta,
            device=attribute_args.device,
        )
        print(f"Saving attributions to {attribute_args.save_path}")
        out.save(attribute_args.save_path, overwrite=True)
