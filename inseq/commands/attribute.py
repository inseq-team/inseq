from typing import Any, List, Optional, Sequence

from dataclasses import dataclass, field

import torch

from .. import AttributionModel, FeatureAttributionOutput, list_feature_attribution_methods, load_model
from .base import BaseCLICommand


@dataclass
class AttributeBaseArgs:
    model_name_or_path: str = field(
        metadata={"alias": "-m", "help": "The name or path of the model on which attribution is performed."},
    )
    attribution_method: Optional[str] = field(
        default="integrated_gradients",
        metadata={
            "alias": "-am",
            "help": "The attribution method used to perform feature attribution.",
            "choices": list_feature_attribution_methods(),
        },
    )
    do_prefix_attribution: bool = field(
        default=False,
        metadata={
            "alias": "-pa",
            "help": "Performs the attribution procedure including the generated prefix at every step.",
        },
    )
    output_step_probabilities: bool = field(
        default=False, metadata={"alias": "-sp", "help": "Adds step decoding probabilities to the attribution output."}
    )
    output_step_attributions: bool = field(
        default=False, metadata={"alias": "-sa", "help": "Adds step-level feature attributions to the output."}
    )
    include_eos_baseline: bool = field(
        default=False,
        metadata={
            "alias": "-eos",
            "help": "Whether the EOS token should be included in the baseline, used for some attribution methods.",
        },
    )
    n_approximation_steps: Optional[int] = field(
        default=100,
        metadata={"alias": "-ns", "help": "Number of approximation steps, used for some attribution methods."},
    )
    return_convergence_delta: bool = field(
        default=False,
        metadata={
            "alias": "-cd",
            "help": "Returns the convergence delta of the approximation, used for some attribution methods.",
        },
    )
    batch_size: int = field(
        default=8,
        metadata={
            "alias": "-bs",
            "help": "The batch size used for the attribution computation. By default, no batching is performed.",
        },
    )
    attribution_batch_size: Optional[int] = field(
        default=50,
        metadata={
            "alias": "-abs",
            "help": "The internal batch size used by the attribution method, used for some attribution methods.",
        },
    )
    device: str = field(
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        metadata={"alias": "-dev", "help": "The device used for inference with Pytorch. Multi-GPU is not supported."},
    )
    hide_attributions: bool = field(
        default=False,
        metadata={
            "alias": "-noshow",
            "help": "If specified, the attribution visualization are not shown in the output.",
        },
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"alias": "-o", "help": "Path where the attribution output should be saved in JSON format."},
    )


@dataclass
class AttributeArgs(AttributeBaseArgs):
    input_texts: List[str] = field(
        default=None, metadata={"alias": "-i", "help": "One or more input texts used for generation."}
    )
    generated_texts: Optional[List[str]] = field(
        default=None,
        metadata={
            "alias": "-gen",
            "help": "If specified, constrains the decoding procedure to the specified outputs.",
        },
    )

    def __post_init__(self):
        if self.input_texts is None:
            raise RuntimeError("Input texts must be specified.")
        if isinstance(self.input_texts, str):
            self.input_texts = [t for t in self.input_texts]
        if isinstance(self.generated_texts, str):
            self.generated_texts = [t for t in self.generated_texts]


def batched_attribute(
    args: AttributeBaseArgs,
    model: AttributionModel,
    input_texts: List[str],
    generated_texts: Optional[List[str]] = None,
) -> FeatureAttributionOutput:
    def get_batched(bs: Optional[int], seq: Sequence[Any]) -> List[List[Any]]:
        if bs is None:
            return [seq]
        return [seq[i : i + bs] for i in range(0, len(seq), bs)]  # noqa

    batched_inputs = get_batched(args.batch_size, input_texts)
    batched_generated_texts = [None for _ in batched_inputs]
    if generated_texts is not None:
        batched_generated_texts = get_batched(args.batch_size, generated_texts)
        assert len(batched_inputs) == len(batched_generated_texts)
    outputs = []
    for i, (in_batch, gen_batch) in enumerate(zip(batched_inputs, batched_generated_texts)):
        print(f"Processing batch {i + 1} of {len(batched_inputs)}...")
        outputs.append(
            model.attribute(
                in_batch,
                gen_batch,
                attribute_target=args.do_prefix_attribution,
                output_step_probabilities=args.output_step_probabilities,
                output_step_attributions=args.output_step_attributions,
                include_eos_baseline=args.include_eos_baseline,
                n_steps=args.n_approximation_steps,
                internal_batch_size=args.attribution_batch_size,
                return_convergence_delta=args.return_convergence_delta,
                device=args.device,
            )
        )
    return FeatureAttributionOutput.merge_attributions(outputs)


def attribute(input_texts, generated_texts, args: AttributeBaseArgs):
    model = load_model(args.model_name_or_path, attribution_method=args.attribution_method)
    out = batched_attribute(args, model, input_texts, generated_texts)
    if not args.hide_attributions:
        out.show()
    if args.save_path:
        print(f"Saving attributions to {args.save_path}")
        out.save(args.save_path, overwrite=True)


class AttributeCommand(BaseCLICommand):
    _name = "attribute"
    _help = "Perform feature attribution on one or multiple sentences"
    _dataclasses = AttributeArgs

    def run(args: AttributeArgs):
        attribute(args.input_texts, args.generated_texts, args)
