from typing import List, NoReturn, Optional

import torch
import typer

import inseq


app = typer.Typer(
    name="attribute",
    help="Perform attribution on a sequence-to-sequence model",
)


@app.command("text")
def attribute_text(
    model: str = typer.Option(
        ...,
        "-m",
        "--model",
        case_sensitive=False,
        help="Model name in the 🤗 Hub or path to folder containing local model files.",
    ),
    source_texts: List[str] = typer.Option(
        ...,
        "-s",
        "--source_texts",
        case_sensitive=False,
        help="Source texts to process for performing attribution.",
    ),
    attribution: str = typer.Option(
        "integrated_gradients",
        "-a",
        "--attribution_method",
        case_sensitive=False,
        help="Attribution method to use.",
        autocompletion=lambda: inseq.list_feature_attribution_methods(),
    ),
    target_texts: Optional[List[str]] = typer.Option(
        None,
        "-t",
        "--target_texts",
        case_sensitive=False,
        help="Target texts for performing constrained decoding.",
    ),
    output_attributions_path: Optional[str] = typer.Option(
        None,
        "-o",
        "--output_attributions",
        case_sensitive=False,
        help="Path to save attribution scores.",
    ),
    attribute_target: bool = typer.Option(
        False,
        "--attribute_target",
        case_sensitive=False,
        help="Whether to attribute the target prefix alongisde the source context.",
    ),
    output_step_probabilities: bool = typer.Option(
        False,
        "--output_step_probabilities",
        case_sensitive=False,
        help="Output step probabilities alongside attribution scores.",
    ),
    output_step_attributions: bool = typer.Option(
        False,
        "--output_step_attributions",
        case_sensitive=False,
        help="Output step attributions alongside sequence attributions.",
    ),
    include_eos_baseline: bool = typer.Option(
        False,
        "--include_eos_baseline",
        case_sensitive=False,
        help="Whether the EOS token should be included in the baseline for attribution scores.",
    ),
    n_steps: int = typer.Option(
        300,
        "--n_steps",
        case_sensitive=False,
        help="Number of steps for the attribution, used only for some attribution methods.",
    ),
    return_convergence_delta: bool = typer.Option(
        False,
        "--return_convergence_delta",
        case_sensitive=False,
        help="Whether to return the convergence delta for the attribution, used only for some attribution methods.",
    ),
    internal_batch_size: int = typer.Option(
        100,
        "--internal_batch_size",
        case_sensitive=False,
        help="Batch size for the attribution, used only for some attribution methods.",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        case_sensitive=False,
        help="Device to use for the attribution",
    ),
) -> NoReturn:
    """Perform attribution for the given text using the given model."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not source_texts:
        source_texts = None
    if not target_texts:
        target_texts = None
    model = inseq.load_model(model, attribution_method=attribution)
    out = model.attribute(
        source_texts,
        target_texts,
        attribute_target=attribute_target,
        output_step_probabilities=output_step_probabilities,
        output_step_attributions=output_step_attributions,
        include_eos_baseline=include_eos_baseline,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=return_convergence_delta,
        device=device,
    )
    out.show()
    if output_attributions_path:
        typer.echo(f"Saving attributions to {output_attributions_path}")
        out.save(output_attributions_path, overwrite=True)
