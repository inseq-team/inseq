import dataclasses
import textwrap
import typing


def command_args_docstring(cls):
    """
    A decorator that automatically generates a Google-style docstring for a dataclass.
    """
    docstring = f"{cls.__name__}\n\n"
    field_names = dataclasses.fields(cls)
    resolved_hints = typing.get_type_hints(cls)
    resolved_field_types = {name: resolved_hints[name] for name in field_names}
    if field_names:
        docstring += "**Attributes:**\n"
        for field in field_names:
            field_type = resolved_field_types[field]
            field_help = field.metadata.get("help", "")
            docstring += textwrap.dedent(
                f"""
            **{field.name}** (``{field_type}``): {field_help}
            """
            )
    cls.__doc__ = docstring
    return cls
