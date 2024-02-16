# How to contribute

## Dependencies

We use [`uv`](https://github.com/astral-sh/uv) to manage Inseq dependencies.
If you dont have `uv`, you should install with `make uv-download`.

To install dependencies and prepare [`pre-commit`](https://pre-commit.com/) hooks you would need to run `install` command:

```bash
make install

or

make install-dev
```

To activate your `virtualenv` run `make uv-activate`.

## Codestyle

After installation you may execute code formatting.

```bash
make lint
```

### Checks

Many checks are configured for this project. Command `make check-style` will check style with `ruff`.
The `make check-safety` command will look at the security of your code.

Comand `make lint` applies all checks.

### Before submitting

Before submitting your code please do the following steps:

1. Add any changes you want
2. Add tests for the new changes
3. Edit documentation if you have changed something significant
4. Run `make fix-style` to format your changes.
5. Run `make lint` to ensure that types and security are okay.

## Other help

You can contribute by spreading a word about this library. It would also be a huge contribution to write a short article on how you are using this project. You can also share your best practices with us.
