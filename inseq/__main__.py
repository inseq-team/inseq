import logging

import typer

from .commands import attribute_cmd


app = typer.Typer(
    name="inseq",
    help="Interpretability for Sequence-to-sequence models 🔍",
)
app.add_typer(attribute_cmd, name="attribute")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    app()
