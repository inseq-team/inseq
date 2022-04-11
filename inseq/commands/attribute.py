from typing import List, Optional

from dataclasses import dataclass, field

import torch

from .. import list_feature_attribution_methods, load_model
from .base import BaseCLICommand


@dataclass
class AttributeArgs:
    model_name_or_path: str = field(
        metadata={"help": "The name or path of the model on which attribution is performed."},
    )
    input_texts: List[str] = field(metadata={"help": "One or more input texts used for generation."})
    attribution_method: Optional[str] = field(
        default="integrated_gradients",
        metadata={
            "help": "The attribution method used to perform feature attribution.",
            "choices": list_feature_attribution_methods(),
        },
    )
    generated_texts: Optional[List[str]] = field(
        default=None, metadata={"help": "If specified, constrains the decoding procedure to the specified outputs."}
    )
    save_path: Optional[str] = field(
        default=None, metadata={"help": "Path where the attribution output should be saved in JSON format."}
    )
    do_prefix_attribution: bool = field(
        default=False,
        metadata={"help": "Performs the attribution procedure including the generated prefix at every step."},
    )
    output_step_probabilities: bool = field(
        default=False, metadata={"help": "Adds step decoding probabilities to the attribution output."}
    )
    output_step_attributions: bool = field(
        default=False, metadata={"help": "Adds step-level feature attributions to the output."}
    )
    include_eos_baseline: bool = field(
        default=False,
        metadata={
            "help": "Whether the EOS token should be included in the baseline, used for some attribution methods."
        },
    )
    n_approximation_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of approximation steps, used for some attribution methods."}
    )
    return_convergence_delta: bool = field(
        default=False,
        metadata={"help": "Returns the convergence delta of the approximation, used for some attribution methods."},
    )
    attribution_batch_size: Optional[int] = field(
        default=50,
        metadata={
            "help": "The internal batch size used by the attribution method, used for some attribution methods."
        },
    )
    device: str = field(
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        metadata={"help": "The device used for inference with Pytorch. Multi-GPU is not supported."},
    )

    def __post_init__(self):
        if isinstance(self.input_texts, str):
            self.input_texts = [t for t in self.input_texts]
        if isinstance(self.generated_texts, str):
            self.generated_texts = [t for t in self.generated_texts]


class AttributeCommand(BaseCLICommand):
    _name = "attribute"
    _help = "Perform feature attribution on one or multiple sentences"
    _dataclasses = AttributeArgs

    def run(args: AttributeArgs):
        model = load_model(args.model_name_or_path, attribution_method=args.attribution_method)
        out = model.attribute(
            args.input_texts,
            args.generated_texts,
            attribute_target=args.do_prefix_attribution,
            output_step_probabilities=args.output_step_probabilities,
            output_step_attributions=args.output_step_attributions,
            include_eos_baseline=args.include_eos_baseline,
            n_steps=args.n_approximation_steps,
            internal_batch_size=args.attribution_batch_size,
            return_convergence_delta=args.return_convergence_delta,
            device=args.device,
        )
        out.show()
        if args.save_path:
            print(f"Saving attributions to {args.save_path}")
            out.save(args.save_path, overwrite=True)
