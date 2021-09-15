# type: ignore[attr-defined]
from typing import NoReturn

import logging

import typer
from rich import print

from amseq import AttributionModel

app = typer.Typer(
    name="amseq",
    help="Attribution methods for sequence-to-sequence transformer models 🔍",
    add_completion=False,
)


@app.command(name="attribute")
def main(
    model: str = typer.Option(
        None,
        "-m",
        "--model",
        case_sensitive=False,
        help="Model name in the 🤗 Hub or path to folder containing local model files.",
    ),
    text: str = typer.Option(
        None,
        "-t",
        "--text",
        case_sensitive=False,
        help="Text to process for performing attribution",
    ),
    n_steps: int = typer.Option(
        50,
        "-s",
        "--steps",
        case_sensitive=False,
        help="Number of steps for the integrated gradient method",
    ),
    batch_size: int = typer.Option(
        50,
        "-b",
        "--batch-size",
        case_sensitive=False,
        help="Batch size for the integrated gradient method",
    ),
) -> NoReturn:
    """Perform attribution for the given text using the given model."""
    model = AttributionModel(model)
    print(f"\n{model.attribute(text, n_steps, batch_size)}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    app()
