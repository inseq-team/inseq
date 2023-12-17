import logging
from dataclasses import dataclass
from typing import List, Optional

from .. import list_feature_attribution_methods, load_model
from ..utils import cli_arg, get_default_device
from .base import BaseCLICommand


@dataclass
class AttributeBaseArgs:
    model_name_or_path: str = cli_arg(
        aliases=["-m"], help="The name or path of the model on which attribution is performed."
    )
    attribution_method: Optional[str] = cli_arg(
        default="integrated_gradients",
        aliases=["-a"],
        help="The attribution method used to perform feature attribution.",
        choices=list_feature_attribution_methods(),
    )
    do_prefix_attribution: bool = cli_arg(
        default=False,
        help="Performs the attribution procedure including the generated prefix at every step.",
    )
    generate_from_target_prefix: bool = cli_arg(
        default=False,
        help=(
            "Whether the ``generated_texts`` should be used as target prefixes for the generation process. If False,"
            " the ``generated_texts`` are used as full targets. Option only available for encoder-decoder models,"
            " since for decoder-only ones it is sufficient to add prefix to input string. Default: False."
        ),
    )
    step_scores: List[str] = cli_arg(
        default_factory=list, help="Adds the specified step scores to the attribution output."
    )
    output_step_attributions: bool = cli_arg(default=False, help="Adds step-level feature attributions to the output.")
    include_eos_baseline: bool = cli_arg(
        default=False,
        aliases=["--eos"],
        help="Whether the EOS token should be included in the baseline, used for some attribution methods.",
    )
    batch_size: int = cli_arg(
        default=8, aliases=["-bs"], help="The batch size used for the attribution computation. Default: no batching."
    )
    aggregate_output: bool = cli_arg(
        default=False,
        help="If specified, the attribution output is aggregated using its default aggregator before saving.",
    )
    device: str = cli_arg(
        default=get_default_device(),
        aliases=["--dev"],
        help="The device used for inference with Pytorch. Multi-GPU is not supported.",
    )
    hide_attributions: bool = cli_arg(
        default=False,
        aliases=["--hide"],
        help="If specified, the attribution visualization are not shown in the output.",
    )
    save_path: Optional[str] = cli_arg(
        default=None,
        aliases=["-o"],
        help="Path where the attribution output should be saved in JSON format.",
    )
    viz_path: Optional[str] = cli_arg(
        default=None,
        help="Path where the attribution visualization should be saved in HTML format.",
    )
    start_pos: Optional[int] = cli_arg(
        default=None, aliases=["-s"], help="Start position for the attribution. Default: first token"
    )
    end_pos: Optional[int] = cli_arg(
        default=None, aliases=["-e"], help="End position for the attribution. Default: last token"
    )
    attributed_fn: Optional[str] = cli_arg(
        default=None,
        aliases=["-fn"],
        help="The name of the step function used as attribution target. Default: probability.",
    )
    verbose: bool = cli_arg(
        default=False, aliases=["-v"], help="If specified, use INFO as logging level for the attribution."
    )
    very_verbose: bool = cli_arg(
        default=False, aliases=["-vv"], help="If specified, use DEBUG as logging level for the attribution."
    )
    model_kwargs: dict = cli_arg(
        default_factory=dict,
        help="Additional keyword arguments passed to the model constructor in JSON format.",
    )
    tokenizer_kwargs: dict = cli_arg(
        default_factory=dict,
        help="Additional keyword arguments passed to the tokenizer constructor in JSON format.",
    )
    generation_kwargs: dict = cli_arg(
        default_factory=dict,
        help="Additional keyword arguments passed to the generation method in JSON format.",
    )
    attribution_kwargs: dict = cli_arg(
        default_factory=dict,
        help="Additional keyword arguments passed to the attribution method in JSON format.",
    )


@dataclass
class AttributeArgs(AttributeBaseArgs):
    input_texts: List[str] = cli_arg(default=None, aliases=["-i"], help="One or more input texts used for generation.")
    generated_texts: Optional[List[str]] = cli_arg(
        default=None, aliases=["-g"], help="If specified, constrains the decoding procedure to the specified outputs."
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
        model_kwargs=args.model_kwargs,
        tokenizer_kwargs=args.tokenizer_kwargs,
    )
    out = model.attribute(
        input_texts,
        generated_texts,
        batch_size=args.batch_size,
        attribute_target=args.do_prefix_attribution,
        attributed_fn=args.attributed_fn,
        step_scores=args.step_scores,
        output_step_attributions=args.output_step_attributions,
        include_eos_baseline=args.include_eos_baseline,
        device=args.device,
        generation_args=args.generation_kwargs,
        attr_pos_start=args.start_pos,
        attr_pos_end=args.end_pos,
        generate_from_target_prefix=args.generate_from_target_prefix,
        **args.attribution_kwargs,
    )
    if args.viz_path:
        print(f"Saving visualization to {args.viz_path}")
        html = out.show(return_html=True, display=not args.hide_attributions)
        with open(args.viz_path, "w") as f:
            f.write(html)
    else:
        out.show(display=not args.hide_attributions)
    if args.save_path:
        if args.aggregate_output:
            out = out.aggregate()
        print(f"Saving {'aggregated ' if args.aggregate_output else ''}attributions to {args.save_path}")
        out.save(args.save_path, overwrite=True)


class AttributeCommand(BaseCLICommand):
    _name = "attribute"
    _help = "Perform feature attribution on one or multiple sentences"
    _dataclasses = AttributeArgs

    def run(args: AttributeArgs):
        attribute(args.input_texts, args.generated_texts, args)
