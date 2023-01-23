"""Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/commands/transformers_cli.py"""
import sys

from ..utils import InseqArgumentParser
from .attribute import AttributeCommand
from .attribute_dataset import AttributeDatasetCommand

COMMANDS = [AttributeCommand, AttributeDatasetCommand]


def main():
    parser = InseqArgumentParser(prog="Inseq CLI tool", usage="inseq <COMMAND> [<ARGS>]")
    command_parser = parser.add_subparsers(title="Inseq CLI command helpers")

    for command_type in COMMANDS:
        command_type.register_subcommand(command_parser)

    args = parser.parse_args()

    if not hasattr(args, "factory_method"):
        parser.print_help()
        sys.exit(1)

    # Run
    command, command_args = args.factory_method(args)
    command.run(command_args)


if __name__ == "__main__":
    main()
