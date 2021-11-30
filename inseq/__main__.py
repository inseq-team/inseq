from typing import List, NoReturn, Optional

import logging

import typer

import inseq
from inseq.utils.misc import isnotebook


app = typer.Typer(
    name="inseq",
    help="Interpretability for Sequence-to-sequence models 🔍",
    add_completion=False,
)


@app.command()
def attribute(
    model: str = typer.Option(
        ...,
        "-m",
        "--model",
        case_sensitive=False,
        help="Model name in the 🤗 Hub or path to folder containing local model files.",
    ),
    attribution: str = typer.Option(
        "integrated_gradients",
        "-a",
        "--attribution",
        case_sensitive=False,
        help="Attribution method to use.",
    ),
    texts: List[str] = typer.Option(
        ...,
        "-t",
        "--texts",
        case_sensitive=False,
        help="Text to process for performing attribution",
    ),
    references: Optional[List[str]] = typer.Option(
        None,
        "-r",
        "--references",
        case_sensitive=False,
        help="Reference texts for computing attribution scores",
    ),
    start_index: Optional[int] = typer.Option(
        None,
        "-s",
        "--start",
        case_sensitive=False,
        help="Start index for computing attribution scores",
    ),
    end_index: Optional[int] = typer.Option(
        None,
        "-e",
        "--end",
        case_sensitive=False,
        help="End index for computing attribution scores",
    ),
) -> NoReturn:
    """Perform attribution for the given text using the given model."""
    model = inseq.load(model, attribution_method=attribution)
    out = model.attribute(texts, references, attr_pos_start=start_index, attr_pos_end=end_index)
    if isnotebook():
        inseq.show_attributions(out)
    else:
        if not isinstance(out, list):
            out = [out]
        for attributions in out:
            print(attributions)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    app()
