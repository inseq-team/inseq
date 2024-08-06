import logging

from ... import FeatureAttributionOutput, load_model
from ..base import BaseCLICommand
from .attribute_args import AttributeExtendedArgs, AttributeWithInputsArgs


def aggregate_attribution_scores(
    out: FeatureAttributionOutput,
    selectors: list[int] | None = None,
    aggregators: list[str] | None = None,
    normalize_attributions: bool = False,
    rescale_attributions: bool = False,
) -> FeatureAttributionOutput:
    if selectors is not None and aggregators is not None:
        for select_idx, aggregator_fn in zip(selectors, aggregators, strict=False):
            out = out.aggregate(
                aggregator=aggregator_fn,
                normalize=normalize_attributions,
                rescale=rescale_attributions,
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
                rescale_attributions=args.rescale_attributions,
            )
        print(f"Saving {'aggregated ' if args.aggregate_output else ''}attributions to {args.save_path}")
        out.save(args.save_path, overwrite=True)


class AttributeCommand(BaseCLICommand):
    _name = "attribute"
    _help = "Perform feature attribution on one or multiple sentences"
    _dataclasses = AttributeWithInputsArgs

    def run(args: AttributeWithInputsArgs):
        attribute(args.input_texts, args.generated_texts, args)
