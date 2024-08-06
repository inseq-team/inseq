from dataclasses import dataclass

from ... import (
    list_aggregation_functions,
    list_aggregators,
    list_feature_attribution_methods,
    list_step_functions,
)
from ...utils import cli_arg, get_default_device
from ..commands_utils import command_args_docstring


@command_args_docstring
@dataclass
class AttributeBaseArgs:
    model_name_or_path: str = cli_arg(
        default=None, aliases=["-m"], help="The name or path of the model on which attribution is performed."
    )
    attribution_method: str | None = cli_arg(
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
    attributed_fn: str | None = cli_arg(
        default=None,
        aliases=["-fn"],
        choices=list_step_functions(),
        help=(
            "The attribution target used for the attribution method. Default: ``probability``. If a"
            " step function requiring additional arguments is used (e.g. ``contrast_prob_diff``), they should be"
            " specified using the ``attribution_kwargs`` argument."
        ),
    )
    attribution_selectors: list[int] | None = cli_arg(
        default=None,
        help=(
            "The indices of the attribution scores to be used for the attribution aggregation. If specified, the"
            " aggregation function is applied only to the selected scores, and the other scores are discarded."
            " If not specified, the aggregation function is applied to all the scores."
        ),
    )
    attribution_aggregators: list[str] = cli_arg(
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
    rescale_attributions: bool = cli_arg(
        default=False,
        help=(
            "Whether to rescale the attribution scores for each context. If ``True``, the attribution scores "
            "for each context are rescaled to sum up to the number of tokens in the input, providing an absolute"
            " notion of input salience."
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


@command_args_docstring
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
    step_scores: list[str] = cli_arg(
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
    save_path: str | None = cli_arg(
        default=None,
        aliases=["-o"],
        help="Path where the attribution output should be saved in JSON format.",
    )
    viz_path: str | None = cli_arg(
        default=None,
        help="Path where the attribution visualization should be saved in HTML format.",
    )
    start_pos: int | None = cli_arg(
        default=None, aliases=["-s"], help="Start position for the attribution. Default: first token"
    )
    end_pos: int | None = cli_arg(
        default=None, aliases=["-e"], help="End position for the attribution. Default: last token"
    )
    verbose: bool = cli_arg(
        default=False, aliases=["-v"], help="If specified, use INFO as logging level for the attribution."
    )
    very_verbose: bool = cli_arg(
        default=False, aliases=["-vv"], help="If specified, use DEBUG as logging level for the attribution."
    )


@command_args_docstring
@dataclass
class AttributeWithInputsArgs(AttributeExtendedArgs):
    input_texts: list[str] = cli_arg(default=None, aliases=["-i"], help="One or more input texts used for generation.")
    generated_texts: list[str] | None = cli_arg(
        default=None, aliases=["-g"], help="If specified, constrains the decoding procedure to the specified outputs."
    )

    def __post_init__(self):
        if self.input_texts is None:
            raise RuntimeError("Input texts must be specified.")
        if isinstance(self.input_texts, str):
            self.input_texts = list(self.input_texts)
        if isinstance(self.generated_texts, str):
            self.generated_texts = list(self.generated_texts)
