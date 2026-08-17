"""Public command-line interface for generating and verifying datasets."""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from typer.core import TyperGroup

from spatialcf.generation import generate_dataset, inspect_dataset, verify_dataset


def _write_error(error: Exception, *, verbose: bool) -> None:
    if verbose:
        traceback.print_exc(file=sys.stderr)
    payload = {
        "error": str(error),
        "error_type": type(error).__name__,
        "request_id": getattr(error, "request_id", None),
        "stage": getattr(error, "stage", None),
        "status": "ERROR",
    }
    typer.echo(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        err=True,
    )


class _PublicGroup(TyperGroup):
    """Route command parsing failures through the public JSON boundary."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        invocation_args = sys.argv[1:] if args is None else args
        verbose = "--verbose" in invocation_args
        try:
            result = super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except Exception as error:  # noqa: BLE001 - top-level CLI boundary
            if not invocation_args and type(error).__name__ == "NoArgsIsHelpError":
                message = str(error)
                if message:
                    typer.echo(message)
                exit_code = getattr(error, "exit_code", 2)
                if standalone_mode:
                    raise SystemExit(exit_code) from None
                return exit_code
            _write_error(error, verbose=verbose)
            if standalone_mode:
                raise SystemExit(2) from None
            return 2
        if standalone_mode and type(result) is int and result != 0:
            raise SystemExit(result)
        return result


app = typer.Typer(
    cls=_PublicGroup,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Generate, verify, and inspect SpatialCF datasets.",
)
_ResultT = TypeVar("_ResultT")


def _run(action: Callable[[], _ResultT], *, verbose: bool) -> _ResultT:
    """Run one facade call behind the CLI's stable JSON error boundary."""

    try:
        return action()
    except Exception as error:  # noqa: BLE001 - public CLI error boundary
        _write_error(error, verbose=verbose)
        raise typer.Exit(code=2) from None


@app.command()
def generate(
    config: Annotated[Path, typer.Option(dir_okay=False)],
    output: Annotated[Path, typer.Option(file_okay=False)],
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Generate and publish a dataset from a TOML configuration."""

    report = _run(lambda: generate_dataset(config, output), verbose=verbose)
    typer.echo(report.model_dump_json(indent=2))


@app.command()
def verify(
    dataset: Path,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Freshly verify a generated dataset and its checksums."""

    report = _run(lambda: verify_dataset(dataset), verbose=verbose)
    typer.echo(report.model_dump_json(indent=2))


@app.command()
def inspect(
    dataset: Path,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Print a compact summary after full dataset verification."""

    summary = _run(lambda: inspect_dataset(dataset), verbose=verbose)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
