import logging
from dataclasses import dataclass
from typing import List, Optional

from .. import (
    FeatureAttributionOutput,
    list_aggregation_functions,
    list_aggregators,
    list_feature_attribution_methods,
    list_step_functions,
    load_model,
)
from ..utils import cli_arg, get_default_device
from .base import BaseCLICommand


@dataclass
class AttributeBaseArgs:
    model_name_or_path: str = cli_arg(
        default=None, aliases=["-m"], help="The name or path of the model on which attribution is performed."
    )
    attribution_method: Optional[str] = cli_arg(
        default="saliency",
        aliases=["-a"],
        help="The attribution method used to perform feature attribution.",
        choices=list_feature_attribution_methods(),
    )
    device: str = cli_arg(
        default=get_default_device(),
        aliases=["--dev"],
        help="The device used for inference with Pytorch. Multi-GPU is not supported.",
    )
    attributed_fn: Optional[str] = cli_arg(
        default=None,
        aliases=["-fn"],
        choices=list_step_functions(),
        help=(
            "The attribution target used for the attribution method. Default: ``probability``. If a"
            " step function requiring additional arguments is used (e.g. ``contrast_prob_diff``), they should be"
            " specified using the ``attribution_kwargs`` argument."
        ),
    )
    attribution_selectors: Optional[List[int]] = cli_arg(
        default=None,
        help=(
            "The indices of the attribution scores to be used for the attribution aggregation. If specified, the"
            " aggregation function is applied only to the selected scores, and the other scores are discarded."
            " If not specified, the aggregation function is applied to all the scores."
        ),
    )
    attribution_aggregators: List[str] = cli_arg(
        default=None,
        help=(
            "The aggregators used to aggregate the attribution scores for each context. The outcome should"
            " produce one score per input token"
        ),
        choices=list_aggregators() + list_aggregation_functions(),
    )
    normalize_attributions: bool = cli_arg(
        default=False,
        help=(
            "Whether to normalize the attribution scores for each context. If ``True``, the attribution scores "
            "for each context are normalized to sum up to 1, providing a relative notion of input salience."
        ),
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
class AttributeExtendedArgs(AttributeBaseArgs):
    attribute_target: bool = cli_arg(
        default=False,
        help="Performs the attribution procedure including the generated target prefix at every step.",
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
        default_factory=list,
        help="Adds the specified step scores to the attribution output.",
        choices=list_step_functions(),
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
    verbose: bool = cli_arg(
        default=False, aliases=["-v"], help="If specified, use INFO as logging level for the attribution."
    )
    very_verbose: bool = cli_arg(
        default=False, aliases=["-vv"], help="If specified, use DEBUG as logging level for the attribution."
    )


@dataclass
class AttributeWithInputsArgs(AttributeExtendedArgs):
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


def aggregate_attribution_scores(
    out: FeatureAttributionOutput,
    selectors: Optional[List[int]] = None,
    aggregators: Optional[List[str]] = None,
    normalize_attributions: bool = False,
) -> FeatureAttributionOutput:
    if selectors is not None and aggregators is not None:
        for select_idx, aggregator_fn in zip(selectors, aggregators):
            out = out.aggregate(
                aggregator=aggregator_fn,
                normalize=normalize_attributions,
                select_idx=select_idx,
                do_post_aggregation_checks=False,
            )
    else:
        out = out.aggregate(aggregator=aggregators, normalize=normalize_attributions)
    return out


def attribute(input_texts, generated_texts, args: AttributeExtendedArgs):
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
    # Handle language tag for multilingual models - no need to specify it in generation kwargs
    if "tgt_lang" in args.tokenizer_kwargs and "forced_bos_token_id" not in args.generation_kwargs:
        tgt_lang = args.tokenizer_kwargs["tgt_lang"]
        args.generation_kwargs["forced_bos_token_id"] = model.tokenizer.lang_code_to_id[tgt_lang]

    out = model.attribute(
        input_texts,
        generated_texts,
        batch_size=args.batch_size,
        attribute_target=args.attribute_target,
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
        if args.attribution_aggregators is not None:
            out = aggregate_attribution_scores(
                out=out,
                selectors=args.attribution_selectors,
                aggregators=args.attribution_aggregators,
                normalize_attributions=args.normalize_attributions,
            )
        print(f"Saving {'aggregated ' if args.aggregate_output else ''}attributions to {args.save_path}")
        out.save(args.save_path, overwrite=True)


class AttributeCommand(BaseCLICommand):
    _name = "attribute"
    _help = "Perform feature attribution on one or multiple sentences"
    _dataclasses = AttributeWithInputsArgs

    def run(args: AttributeWithInputsArgs):
        attribute(args.input_texts, args.generated_texts, args)
