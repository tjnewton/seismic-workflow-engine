"""Top-level CLI entry point for the Seismic Workflow Engine.

Usage::

    swe --help
    swe --version
    swe run pipeline.yml
    swe run --dry-run pipeline.yml
    swe pipeline template full
    swe pipeline list
"""

from __future__ import annotations

import logging

import click

import swe
from swe.cli.commands import associate, detect, download, locate, noise, pick, plot
from swe.pipeline import Pipeline

logger = logging.getLogger(__name__)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=swe.__version__, prog_name="swe")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v for INFO, -vv for DEBUG).",
)
def cli(*, verbose: int) -> None:
    """Seismic Workflow Engine - modular seismic analysis toolkit."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--skip", multiple=True, help="Step names to skip.")
@click.option(
    "--run-dir",
    default=".swe_runs",
    help="Base directory for run state.",
)
@click.option(
    "--resume",
    default=None,
    help="Run ID to resume (or 'latest').",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable output caching.",
)
@click.option(
    "--parallel",
    is_flag=True,
    default=False,
    help="Run independent steps concurrently.",
)
@click.option(
    "--show-provenance",
    default=None,
    help="Show provenance for a run ID (or 'latest') and exit.",
)
@click.option(
    "--propagate-uncertainty/--no-propagate-uncertainty",
    default=False,
    help="Enable uncertainty propagation between pipeline steps.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate the pipeline without executing it.",
)
def run(
    config_file: str,
    skip: tuple[str, ...],
    run_dir: str,
    resume: str | None,
    no_cache: bool,
    parallel: bool,
    show_provenance: str | None,
    propagate_uncertainty: bool,
    dry_run: bool,
) -> None:
    """Execute a YAML pipeline configuration."""
    if show_provenance is not None:
        from pathlib import Path

        from swe.pipeline.provenance import format_provenance, load_provenance
        from swe.pipeline.run_context import RunContext

        base = Path(run_dir)
        ctx = RunContext.load(base, show_provenance)
        records = load_provenance(ctx.run_dir)
        click.echo(format_provenance(records))
        return

    try:
        pipeline = Pipeline.from_yaml(config_file)
    except ValueError as exc:  # a malformed pipeline: an error line, not a traceback
        raise click.ClickException(str(exc)) from exc

    if dry_run:
        from swe.pipeline.validation import validate_pipeline

        errors, warnings = validate_pipeline(pipeline)
        if warnings:
            for w in warnings:
                click.echo(f"  WARNING: {w}")
        if errors:
            for e in errors:
                click.echo(f"  ERROR: {e}")
            raise click.ClickException(f"Dry run failed with {len(errors)} error(s)")
        click.echo("Dry run passed: pipeline is valid.")
        return

    if propagate_uncertainty:
        for step in pipeline.steps:
            step.propagate_uncertainty = True
        logger.info("Uncertainty propagation enabled for all steps")
    pipeline.execute(
        skip=set(skip),
        run_dir=run_dir,
        resume=resume,
        no_cache=no_cache,
        parallel=parallel,
    )


# --- Pipeline management subgroup ---


@cli.group()
def pipeline() -> None:
    """Manage pipeline templates and configurations."""


@pipeline.command("template")
@click.argument(
    "workflow",
    type=click.Choice(
        ["full", "detection", "stress", "spectral", "magnitude", "uncertainty", "roughness"],
        case_sensitive=False,
    ),
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path (defaults to '<workflow>_pipeline.yml').",
)
def pipeline_template(workflow: str, output: str | None) -> None:
    """Scaffold a starter YAML pipeline configuration.

    Generates a well-commented template for the chosen workflow that
    you can customize for your data.
    """
    from pathlib import Path

    from swe.pipeline.templates import get_template

    content = get_template(workflow)

    if output is None:
        output = f"{workflow}_pipeline.yml"

    out_path = Path(output)
    if out_path.exists():
        raise click.ClickException(
            f"File already exists: {out_path}. Use --output to specify a different path."
        )

    out_path.write_text(content)
    click.echo(f"Created {out_path}")


@pipeline.command("list")
def pipeline_list() -> None:
    """List available pipeline templates."""
    from swe.pipeline.templates import list_templates

    templates = list_templates()
    click.echo("Available pipeline templates:")
    for name in templates:
        click.echo(f"  {name}")
    click.echo("\nUse 'swe pipeline template <name>' to scaffold a config.")


cli.add_command(download)
cli.add_command(noise)
cli.add_command(pick)
cli.add_command(associate)
cli.add_command(detect)
cli.add_command(locate)
cli.add_command(plot)
