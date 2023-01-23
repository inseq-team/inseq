import logging
from dataclasses import dataclass, field
from typing import List, Optional

from .. import list_feature_attribution_methods, load_model
from ..utils import get_default_device
from .base import BaseCLICommand


@dataclass
class AttributeBaseArgs:
    model_name_or_path: str = field(
        metadata={"alias": "-m", "help": "The name or path of the model on which attribution is performed."},
    )
    attribution_method: Optional[str] = field(
        default="integrated_gradients",
        metadata={
            "alias": "-a",
            "help": "The attribution method used to perform feature attribution.",
            "choices": list_feature_attribution_methods(),
        },
    )
    do_prefix_attribution: bool = field(
        default=False,
        metadata={
            "help": "Performs the attribution procedure including the generated prefix at every step.",
        },
    )
    step_scores: List[str] = field(
        default_factory=list, metadata={"help": "Adds step scores to the attribution output."}
    )
    output_step_attributions: bool = field(
        default=False, metadata={"help": "Adds step-level feature attributions to the output."}
    )
    include_eos_baseline: bool = field(
        default=False,
        metadata={
            "alias": "--eos",
            "help": "Whether the EOS token should be included in the baseline, used for some attribution methods.",
        },
    )
    n_approximation_steps: Optional[int] = field(
        default=100,
        metadata={"alias": "-n", "help": "Number of approximation steps, used for some attribution methods."},
    )
    return_convergence_delta: bool = field(
        default=False,
        metadata={
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
            "help": "The internal batch size used by the attribution method, used for some attribution methods.",
        },
    )
    device: str = field(
        default=get_default_device(),
        metadata={"alias": "--dev", "help": "The device used for inference with Pytorch. Multi-GPU is not supported."},
    )
    hide_attributions: bool = field(
        default=False,
        metadata={
            "alias": "--hide",
            "help": "If specified, the attribution visualization are not shown in the output.",
        },
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"alias": "-o", "help": "Path where the attribution output should be saved in JSON format."},
    )
    viz_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path where the attribution visualization should be saved in HTML format.",
        },
    )
    max_gen_length: Optional[int] = field(
        default=None,
        metadata={"alias": "-l", "help": "Max generation length for model outputs. Default: 512"},
    )
    start_pos: Optional[int] = field(
        default=None,
        metadata={"alias": "-s", "help": "Start position for the attribution. Default: first token"},
    )
    end_pos: Optional[int] = field(
        default=None,
        metadata={"alias": "-e", "help": "End position for the attribution. Default: last token"},
    )
    verbose: bool = field(
        default=False,
        metadata={"alias": "-v", "help": "If specified, use INFO as logging level for the attribution."},
    )
    very_verbose: bool = field(
        default=False,
        metadata={"alias": "-vv", "help": "If specified, use DEBUG as logging level for the attribution."},
    )


@dataclass
class AttributeArgs(AttributeBaseArgs):
    input_texts: List[str] = field(
        default=None, metadata={"alias": "-i", "help": "One or more input texts used for generation."}
    )
    generated_texts: Optional[List[str]] = field(
        default=None,
        metadata={
            "alias": "-g",
            "help": "If specified, constrains the decoding procedure to the specified outputs.",
        },
    )

    def __post_init__(self):
        if self.input_texts is None:
            raise RuntimeError("Input texts must be specified.")
        if isinstance(self.input_texts, str):
            self.input_texts = list(self.input_texts)
        if isinstance(self.generated_texts, str):
            self.generated_texts = list(self.generated_texts)


def attribute(input_texts, generated_texts, args: AttributeBaseArgs):
    if args.very_verbose:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    model = load_model(
        args.model_name_or_path,
        attribution_method=args.attribution_method,
        device=args.device,
    )
    out = model.attribute(
        input_texts,
        generated_texts,
        batch_size=args.batch_size,
        attribute_target=args.do_prefix_attribution,
        step_scores=args.step_scores,
        output_step_attributions=args.output_step_attributions,
        include_eos_baseline=args.include_eos_baseline,
        n_steps=args.n_approximation_steps,
        internal_batch_size=args.attribution_batch_size,
        return_convergence_delta=args.return_convergence_delta,
        device=args.device,
        generation_args={"max_new_tokens": args.max_gen_length},
        attr_pos_start=args.start_pos,
        attr_pos_end=args.end_pos,
    )
    if args.viz_path:
        print(f"Saving visualization to {args.viz_path}")
        html = out.show(return_html=True, display=not args.hide_attributions)
        with open(args.viz_path, "w") as f:
            f.write(html)
    else:
        out.show(display=not args.hide_attributions)
    if args.save_path:
        print(f"Saving attributions to {args.save_path}")
        out.save(args.save_path, overwrite=True)


class AttributeCommand(BaseCLICommand):
    _name = "attribute"
    _help = "Perform feature attribution on one or multiple sentences"
    _dataclasses = AttributeArgs

    def run(args: AttributeArgs):
        attribute(args.input_texts, args.generated_texts, args)
