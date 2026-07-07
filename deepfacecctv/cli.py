"""DeepFaceCCTV CLI — thin command layer over existing business logic."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Annotated, Optional


import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deepfacecctv import __app_name__, __version__
from deepfacecctv.config import Config, load_config

app = typer.Typer(
    name=__app_name__,
    help="AI-powered CCTV face identification pipeline",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


# ── Shared options ───────────────────────────────────────────────────────────

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to configuration file", exists=False, dir_okay=False),
]

CameraOption = Annotated[
    Optional[str],
    typer.Option(
        "--camera", "-s", help="Video source: camera index (0), file path, or RTSP/HTTP URL"
    ),
]

HeadlessOption = Annotated[
    bool,
    typer.Option("--headless", help="Run without GUI window (background mode)"),
]

PortOption = Annotated[
    Optional[int],
    typer.Option("--port", "-p", help="Server port", min=1024, max=65535),
]

HostOption = Annotated[
    Optional[str],
    typer.Option("--host", "-h", help="Server bind address"),
]


# ── Version callback ─────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold green]{__app_name__}[/bold green] version [cyan]{__version__}[/cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """DeepFaceCCTV — AI-powered CCTV face identification pipeline."""
    pass


# ── Command: run ─────────────────────────────────────────────────────────────


@app.command()
def run(
    camera: CameraOption = None,
    config: ConfigOption = Path("config.json"),
    headless: HeadlessOption = False,
) -> None:
    """
    Start the face detection pipeline.

    Examples:
        $ deepfacecctv run                    # Use config.json source
        $ deepfacecctv run --camera 0         # Use webcam
        $ deepfacecctv run -s video.mp4       # Use video file
        $ deepfacecctv run -s rtsp://...     # Use RTSP stream
        $ deepfacecctv run --headless        # Background mode
    """
    cfg = load_config(config)

    if camera is not None:
        cfg.source = camera
    if headless:
        cfg.headless = True

    if not cfg.source:
        console.print(
            Panel.fit(
                "[bold red]Error:[/bold red] No source specified.\n"
                "Use [cyan]--camera[/cyan] or set [cyan]source[/cyan] in config.json",
                title="Missing Source",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    _show_config_summary(cfg)

    console.print(f"\n[bold]Starting pipeline...[/bold] Press [yellow]Ctrl+C[/yellow] to stop.\n")

    try:
        from deepfacecctv.pipeline_adapter import run_pipeline

        run_pipeline(cfg)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown requested by user.[/yellow]")
        raise typer.Exit(0)
    except Exception as exc:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {exc}")
        raise typer.Exit(1)


# ── Command: dashboard ─────────────────────────────────────────────────────────


@app.command()
def dashboard(
    host: HostOption = None,
    port: PortOption = None,
    config: ConfigOption = Path("config.json"),
) -> None:
    """Start the Flask dashboard (web UI) independently."""
    cfg = load_config(config)

    if host is not None:
        cfg.flask_host = host
    if port is not None:
        cfg.flask_port = port

    # Determine display URL
    display_host = "localhost" if cfg.flask_host == "0.0.0.0" else cfg.flask_host

    console.print(
        Panel.fit(
            f"[bold]Dashboard[/bold] starting...\n\n"
            f"Local access:   [cyan]http://localhost:{cfg.flask_port}[/cyan]\n"
            f"                [cyan]http://127.0.0.1:{cfg.flask_port}[/cyan]\n"
            f"Network access: [cyan]http://YOUR_IP:{cfg.flask_port}[/cyan]\n\n"
            f"Database: [dim]{cfg.db_path}[/dim]\n"
            f"Detections: [dim]{cfg.output_db}[/dim]",
            title="DeepFaceCCTV Dashboard",
            border_style="blue",
        )
    )

    try:
        from deepfacecctv.pipeline_adapter import start_dashboard

        start_dashboard(cfg)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")
    except Exception as exc:
        console.print(f"\n[bold red]Dashboard failed:[/bold red] {exc}")
        raise typer.Exit(1)


# ── Command group: status ────────────────────────────────────────────────────
@app.command()
def status(
    host: Annotated[
        Optional[str],
        typer.Option("--host", "-h", help="Dashboard host"),
    ] = "localhost",
    port: Annotated[
        Optional[int],
        typer.Option("--port", "-p", help="Dashboard port"),
    ] = 5002,
) -> None:
    """
    Check dashboard health/status.

    Verifies the dashboard server is running and database is accessible.
    """
    import urllib.request
    import json

    url = f"http://{host}:{port}/health"

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read())

            if data.get("status") == "ok":
                console.print(
                    Panel.fit(
                        f"[bold green]✓ Dashboard is healthy[/bold green]\n\n"
                        f"Database: [green]{data['db']}[/green]\n"
                        f"Unique people: [cyan]{data['unique_people']}[/cyan]\n"
                        f"Total detections: [dim]{data['total_detections']}[/dim]\n"
                        f"Timestamp: {data['timestamp']}",
                        title="Status Check",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel.fit(
                        f"[bold red]✗ Dashboard error[/bold red]\n\n"
                        f"Status: {data['status']}\n"
                        f"Error: {data.get('db', 'unknown')}",
                        title="Status Check",
                        border_style="red",
                    )
                )
                raise typer.Exit(1)

    except Exception as e:
        console.print(
            Panel.fit(
                f"[bold red]✗ Cannot connect to dashboard[/bold red]\n\n"
                f"URL: [cyan]{url}[/cyan]\n"
                f"Error: {e}\n\n"
                f"Is the dashboard running?\n"
                f"Start it with: [cyan]deepfacecctv dashboard[/cyan]",
                title="Connection Failed",
                border_style="red",
            )
        )
        raise typer.Exit(1)


# ── Command group: config ────────────────────────────────────────────────────

config_app = typer.Typer(help="Manage configuration", no_args_is_help=True)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(config: ConfigOption = Path("config.json")) -> None:
    """Display current configuration values."""
    cfg = load_config(config)

    table = Table(title=f"Configuration: [cyan]{config}[/cyan]", show_header=True)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")

    table.add_row("", "", "[bold]Paths[/bold]")
    table.add_row("Config file", str(config), "Configuration file path")
    table.add_row("Database", cfg.db_path, "ChromaDB face gallery")
    table.add_row("Detections DB", cfg.output_db, "SQLite detection logs")
    table.add_row("Crops dir", cfg.crops_dir, "Face crop images")
    table.add_row("YuNet model", cfg.yunet_model, "Face detection ONNX")
    table.add_row("SFace model", cfg.sface_model, "Face recognition ONNX")

    table.add_row("", "", "[bold]Source[/bold]")
    table.add_row("Source", cfg.source or "(not set)", cfg.source_label)
    table.add_row("Transport", cfg.transport, "RTSP protocol")

    table.add_row("", "", "[bold]Pipeline[/bold]")
    table.add_row("Headless", str(cfg.headless), "No GUI window")
    table.add_row("Confidence", str(cfg.det_confidence), "Detection threshold")
    table.add_row("Gallery refresh", f"{cfg.gallery_refresh_sec}s", "Cache interval")

    table.add_row("", "", "[bold]Dashboard[/bold]")
    table.add_row("Host", cfg.flask_host, "Bind address")
    table.add_row("Port", str(cfg.flask_port), "HTTP port")

    console.print(table)

# ── Command group: gallery ───────────────────────────────────────────────────

gallery_app = typer.Typer(
    help="Manage face recognition gallery (ChromaDB)",
    no_args_is_help=True,
)
dataset_app = typer.Typer(
    help="Build and manage face datasets for gallery enrollment",
    no_args_is_help=True,
)
app.add_typer(gallery_app, name="gallery")
# Add this line (if missing)
app.add_typer(dataset_app, name="dataset")


@gallery_app.command("list")
def gallery_list(
    config: ConfigOption = Path("config.json"),
) -> None:
    """List all enrolled identities in the gallery."""
    cfg = load_config(config)

    try:
        from deepfacecctv.pipeline_adapter import list_gallery_identities

        identities = list_gallery_identities(
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
        )

        if not identities:
            console.print("[yellow]Gallery is empty.[/yellow]")
            return

        table = Table(
            title=f"Gallery: {cfg.collection_name} ({len(identities)} identities)",
            show_header=True,
        )
        table.add_column("Name", style="cyan")
        table.add_column("Embeddings", style="green", justify="right")
        table.add_column("Last Updated", style="dim")

        for name, count, updated in identities:
            table.add_row(name, str(count), updated or "unknown")

        console.print(table)

    except Exception as exc:
        console.print(f"[bold red]Failed to list gallery:[/bold red] {exc}")
        raise typer.Exit(1)


@gallery_app.command("info")
def gallery_info(
    config: ConfigOption = Path("config.json"),
) -> None:
    """Show gallery statistics."""
    cfg = load_config(config)

    try:
        from deepfacecctv.pipeline_adapter import get_gallery_info

        info = get_gallery_info(
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
        )

        table = Table(title=f"Gallery Info: {cfg.collection_name}", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Database path", cfg.db_path)
        table.add_row("Collection", cfg.collection_name)
        table.add_row("Total identities", str(info.get("identities", 0)))
        table.add_row("Total embeddings", str(info.get("embeddings", 0)))
        table.add_row("Database size", info.get("size", "unknown"))

        console.print(table)

    except Exception as exc:
        console.print(f"[bold red]Failed:[/bold red] {exc}")
        raise typer.Exit(1)


@gallery_app.command("enroll")
def gallery_enroll(
    name: Annotated[
        str,
        typer.Argument(..., help="Person name/identifier"),
    ],
    images: Annotated[
        list[Path],
        typer.Argument(
            ...,
            help="Image file(s) to enroll",
            exists=True,
            dir_okay=False,
            file_okay=True,
        ),
    ],
    config: ConfigOption = Path("config.json"),
) -> None:
    """
    Enroll a new person into the gallery.

    Examples:
        $ deepfacecctv gallery enroll "John Doe" photos/john1.jpg photos/john2.jpg
        $ deepfacecctv gallery enroll "Alice" alice.png
    """
    cfg = load_config(config)

    console.print(f"Enrolling [cyan]{name}[/cyan] with {len(images)} image(s)...")

    try:
        from deepfacecctv.pipeline_adapter import enroll_identity

        result = enroll_identity(
            name=name,
            image_paths=images,
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
            yunet_model=cfg.yunet_model,
        )

        console.print(f"\n[green]✓[/green] Enrolled [bold]{name}[/bold]")
        console.print(f"  Embeddings created: {result.get('embeddings', 0)}")
        console.print(f"  Images processed: {result.get('images', 0)}")

    except Exception as exc:
        console.print(f"\n[bold red]Enrollment failed:[/bold red] {exc}")
        raise typer.Exit(1)


@gallery_app.command("delete")
def gallery_delete(
    name: Annotated[
        str,
        typer.Argument(..., help="Person name to remove"),
    ],
    config: ConfigOption = Path("config.json"),
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Remove an identity from the gallery."""
    cfg = load_config(config)

    if not yes:
        confirm = typer.confirm(f"Delete '{name}' from gallery?")
        if not confirm:
            console.print("Cancelled.")
            raise typer.Exit(0)

    try:
        from deepfacecctv.pipeline_adapter import delete_identity

        deleted = delete_identity(
            name=name,
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
        )

        if deleted:
            console.print(f"[green]✓[/green] Deleted '{name}' from gallery.")
        else:
            console.print(f"[yellow]Identity '{name}' not found.[/yellow]")

    except Exception as exc:
        console.print(f"[bold red]Delete failed:[/bold red] {exc}")
        raise typer.Exit(1)


@gallery_app.command("backup")
def gallery_backup(
    config: ConfigOption = Path("config.json"),
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Backup directory"),
    ] = None,
) -> None:
    """Backup gallery to timestamped folder."""
    cfg = load_config(config)

    import shutil
    from datetime import datetime

    src = Path(cfg.db_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = output or Path(f"data/face_db_backup_{timestamp}")

    try:
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            console.print(f"[green]✓[/green] Gallery backed up to: {dst}")
        else:
            console.print(f"[yellow]Source not found:[/yellow] {src}")

    except Exception as exc:
        console.print(f"[bold red]Backup failed:[/bold red] {exc}")
        raise typer.Exit(1)


@gallery_app.command("restore")
def gallery_restore(
    backup: Annotated[
        Path,
        typer.Argument(..., help="Backup directory to restore from", exists=True, dir_okay=True),
    ],
    config: ConfigOption = Path("config.json"),
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation"),
    ] = False,
) -> None:
    """Restore gallery from backup."""
    cfg = load_config(config)

    if not yes:
        confirm = typer.confirm(f"Restore gallery from {backup}? This will overwrite current data!")
        if not confirm:
            console.print("Cancelled.")
            raise typer.Exit(0)

    try:
        import shutil

        dst = Path(cfg.db_path)

        # Remove current and restore
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(backup, dst)

        console.print(f"[green]✓[/green] Gallery restored from: {backup}")

    except Exception as exc:
        console.print(f"[bold red]Restore failed:[/bold red] {exc}")
        raise typer.Exit(1)


# ── Command group: dataset ─────────────────────────────────────────────────

# ── Command group: dataset ─────────────────────────────────────────────────

dataset_app = typer.Typer(
    help="Build and manage face datasets for gallery enrollment",
    no_args_is_help=True,
)
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("build")
def dataset_build(
    input_dir: Annotated[
        Optional[Path],
        typer.Argument(
            help="Directory with face images organized by identity (person_name/img.jpg)",
            exists=False,  # Don't require it to exist
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output dataset directory",
            dir_okay=True,
            file_okay=False,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force rebuild even if gallery exists",
        ),
    ] = False,
    config: ConfigOption = Path("config.json"),
) -> None:
    """
    Build a face dataset from organized images.

    If no input directory is provided, checks if gallery already exists.
    """
    cfg = load_config(config)

    # Check if gallery already exists
    gallery_exists = _check_gallery_exists(cfg.db_path, cfg.collection_name)

    if gallery_exists and not force:
        console.print(
            Panel.fit(
                f"[bold yellow]⚠ Gallery already exists![/bold yellow]\n\n"
                f"Database: [cyan]{cfg.db_path}[/cyan]\n"
                f"Collection: [cyan]{cfg.collection_name}[/cyan]\n\n"
                f"Your gallery already has enrolled faces. Building a new dataset\n"
                f"will add more embeddings to the existing gallery.\n\n"
                f"Use [cyan]--force[/cyan] to proceed anyway, or use [cyan]gallery[/cyan] commands to manage.",
                title="Gallery Exists",
                border_style="yellow",
            )
        )
        raise typer.Exit(0)

    # If no input provided, show error
    if input_dir is None:
        console.print(
            Panel.fit(
                "[bold red]Error:[/bold red] No input directory specified.\n\n"
                "Usage:\n"
                "  [cyan]deepfacecctv dataset build ./my_photos[/cyan]\n\n"
                "Expected structure:\n"
                "  ./my_photos/\n"
                "    person_a/\n"
                "      photo1.jpg\n"
                "      photo2.jpg\n"
                "    person_b/\n"
                "      photo1.jpg",
                title="Missing Input",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    # Validate input exists
    if not input_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Directory not found: {input_dir}")
        raise typer.Exit(1)

    out_dir = output or Path("data/dataset")

    console.print(
        Panel.fit(
            f"[bold]Building dataset[/bold]\n"
            f"Input:  [cyan]{input_dir}[/cyan]\n"
            f"Output: [cyan]{out_dir}[/cyan]",
            title="Dataset Builder",
            border_style="green",
        )
    )

    try:
        from deepfacecctv.pipeline_adapter import build_dataset

        stats = build_dataset(
            input_dir=input_dir,
            output_dir=out_dir,
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
            yunet_model=cfg.yunet_model,
            min_face_size=cfg.min_face_size,
            blur_threshold=cfg.blur_threshold,
        )

        console.print(f"\n[green]✓[/green] Dataset built!")
        console.print(f"  Identities: {stats.get('identities', 0)}")
        console.print(f"  Images processed: {stats.get('images', 0)}")
        console.print(f"  Valid faces: {stats.get('valid_faces', 0)}")
        console.print(f"  Rejected: {stats.get('rejected', 0)}")

    except Exception as exc:
        console.print(f"\n[bold red]Build failed:[/bold red] {exc}")
        raise typer.Exit(1)


@dataset_app.command("info")
def dataset_info(
    dataset: Annotated[
        Optional[Path],
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset image directory",
            dir_okay=True,
            file_okay=False,
        ),
    ] = None,
    config: ConfigOption = Path("config.json"),
) -> None:
    """
    Show dataset or gallery information.

    If no dataset folder exists, shows gallery info instead.
    """
    cfg = load_config(config)

    # If user specified a dataset folder, show that
    if dataset is not None and dataset.exists():
        try:
            from deepfacecctv.pipeline_adapter import get_dataset_info

            info = get_dataset_info(dataset)

            table = Table(title=f"Dataset: {dataset}", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Identities", str(info.get("identities", 0)))
            table.add_row("Total images", str(info.get("total_images", 0)))
            table.add_row("Average per person", str(info.get("avg_images", 0)))
            table.add_row("Dataset size", info.get("size_mb", "0 MB"))

            console.print(table)
            return

        except Exception as exc:
            console.print(f"[bold red]Failed to read dataset:[/bold red] {exc}")
            raise typer.Exit(1)

    # Check if dataset folder exists at default location
    default_dataset = Path("data/dataset")
    if default_dataset.exists() and any(default_dataset.iterdir()):
        try:
            from deepfacecctv.pipeline_adapter import get_dataset_info

            info = get_dataset_info(default_dataset)

            table = Table(title=f"Dataset: {default_dataset}", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Identities", str(info.get("identities", 0)))
            table.add_row("Total images", str(info.get("total_images", 0)))
            table.add_row("Average per person", str(info.get("avg_images", 0)))
            table.add_row("Dataset size", info.get("size_mb", "0 MB"))

            console.print(table)
            console.print(
                f"\n[dim]Tip: Use [cyan]--dataset PATH[/cyan] to check a different folder[/dim]"
            )
            return

        except Exception:
            pass  # Fall through to gallery info

    # No dataset found — show gallery info instead
    console.print("[yellow]No image dataset found.[/yellow] Showing gallery info instead...\n")

    try:
        from deepfacecctv.pipeline_adapter import get_gallery_info

        info = get_gallery_info(cfg.db_path, cfg.collection_name)

        table = Table(title=f"Gallery: {cfg.collection_name}", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Database path", cfg.db_path)
        table.add_row("Collection", cfg.collection_name)
        table.add_row("Total identities", str(info.get("identities", 0)))
        table.add_row("Total embeddings", str(info.get("embeddings", 0)))
        table.add_row("Database size", info.get("size", "unknown"))

        console.print(table)

        if info.get("identities", 0) > 0:
            console.print(
                f"\n[green]✓[/green] Your gallery has {info['identities']} identities ready for recognition."
            )
            console.print(
                f"[dim]To see all identities, run: [cyan]deepfacecctv gallery list[/cyan][/dim]"
            )
        else:
            console.print(f"\n[yellow]Gallery is empty.[/yellow] Build a dataset first:")
            console.print(f"  [cyan]deepfacecctv dataset build ./your_photos[/cyan]")

    except Exception as exc:
        console.print(f"[bold red]Failed to read gallery:[/bold red] {exc}")
        raise typer.Exit(1)


@dataset_app.command("update")
def dataset_update(
    input_dir: Annotated[
        Path,
        typer.Argument(
            ...,
            help="New images directory to merge (use quotes if path has spaces)",
            exists=True,
            dir_okay=True,
            file_okay=False,
            resolve_path=True,
        ),
    ],
    dataset: Annotated[
        Optional[Path],
        typer.Option(
            "--dataset",
            "-d",
            help="Existing dataset directory (default: updates gallery directly)",
            exists=False,  # Don't require it
            dir_okay=True,
            file_okay=False,
        ),
    ] = None,
    config: ConfigOption = Path("config.json"),
) -> None:
    """
    Update existing dataset or gallery with new images.

    If --dataset is not provided, updates the gallery directly.

    Examples:
        $ deepfacecctv dataset update ./new_photos
        $ deepfacecctv dataset update "./my photos" --dataset ./data/dataset
        $ deepfacecctv dataset update ./new_faces  # Updates gallery directly
    """
    cfg = load_config(config)

    # Check if gallery exists
    gallery_exists = _check_gallery_exists(cfg.db_path, cfg.collection_name)

    # If no --dataset specified, update gallery directly
    if dataset is None:
        if not gallery_exists:
            console.print(
                Panel.fit(
                    "[bold red]Error:[/bold red] No gallery found to update.\n\n"
                    "Create a gallery first:\n"
                    "  [cyan]deepfacecctv dataset build ./your_photos[/cyan]\n\n"
                    "Or specify a dataset folder:\n"
                    "  [cyan]deepfacecctv dataset update ./new_photos --dataset ./data/dataset[/cyan]",
                    title="No Gallery",
                    border_style="red",
                )
            )
            raise typer.Exit(1)

        console.print(
            Panel.fit(
                f"[bold]Updating gallery directly[/bold]\n"
                f"New images: [cyan]{input_dir}[/cyan]\n"
                f"Target gallery: [cyan]{cfg.collection_name}[/cyan]",
                title="Gallery Update",
                border_style="yellow",
            )
        )

        try:
            from deepfacecctv.pipeline_adapter import update_gallery

            stats = update_gallery(
                new_images_dir=input_dir,
                db_path=cfg.db_path,
                collection_name=cfg.collection_name,
                yunet_model=cfg.yunet_model,
            )

            console.print(f"\n[green]✓[/green] Gallery updated!")
            console.print(f"  New identities: {stats.get('new', 0)}")
            console.print(f"  Updated identities: {stats.get('updated', 0)}")
            console.print(f"  Images added: {stats.get('added', 0)}")

        except Exception as exc:
            console.print(f"\n[bold red]Update failed:[/bold red] {exc}")
            raise typer.Exit(1)

        return

    # Update specific dataset folder
    if not dataset.exists():
        console.print(f"[bold red]Error:[/bold red] Dataset folder not found: {dataset}")
        console.print(
            "Create it first with:\n  [cyan]deepfacecctv dataset build ./your_photos --output {dataset}[/cyan]"
        )
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold]Updating dataset[/bold]\n"
            f"Existing: [cyan]{dataset}[/cyan]\n"
            f"New:      [cyan]{input_dir}[/cyan]",
            title="Dataset Update",
            border_style="yellow",
        )
    )

    try:
        from deepfacecctv.pipeline_adapter import update_dataset

        stats = update_dataset(
            new_images_dir=input_dir,
            existing_dataset=dataset,
            db_path=cfg.db_path,
            collection_name=cfg.collection_name,
            yunet_model=cfg.yunet_model,
        )

        console.print(f"\n[green]✓[/green] Dataset updated!")
        console.print(f"  New identities: {stats.get('new', 0)}")
        console.print(f"  Updated: {stats.get('updated', 0)}")
        console.print(f"  Images added: {stats.get('added', 0)}")

    except Exception as exc:
        console.print(f"\n[bold red]Update failed:[/bold red] {exc}")
        raise typer.Exit(1)


# ── Helper function ────────────────────────────────────────────────────────


def _check_gallery_exists(db_path: str, collection_name: str) -> bool:
    """Check if gallery has any identities."""
    try:
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from builder.db_ops import get_collection

        collection = get_collection(db_path=db_path, collection_name=collection_name)
        results = collection.get(include=["metadatas"])
        return bool(results and results.get("metadatas") and len(results["metadatas"]) > 0)
    except Exception:
        return False


# ── Command: info ────────────────────────────────────────────────────────────


@app.command()
def info() -> None:
    """Display system and project information."""
    table = Table(title="System Information", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Project", __app_name__)
    table.add_row("Version", __version__)
    table.add_row(
        "Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    table.add_row("Executable", sys.executable)
    table.add_row("Platform", platform.platform())
    table.add_row("OS", f"{platform.system()} {platform.release()}")

    for pkg, attr in [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Flask", "flask"),
        ("ChromaDB", "chromadb"),
    ]:
        try:
            mod = __import__(attr)
            ver = getattr(mod, "__version__", "unknown")
            table.add_row(pkg, ver)
        except ImportError:
            table.add_row(pkg, "[red]not installed[/red]")

    console.print(table)


# ── Helper ─────────────────────────────────────────────────────────────────


def _show_config_summary(cfg: Config) -> None:
    table = Table(title="Pipeline Configuration", show_header=False)
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="green")
    table.add_row("Source", cfg.source_label)
    table.add_row("Headless", str(cfg.headless))
    table.add_row("Database", cfg.db_path)
    table.add_row("Detections DB", cfg.output_db)
    table.add_row("Model", cfg.yunet_model)
    table.add_row("Confidence", str(cfg.det_confidence))
    table.add_row("Dashboard Port", str(cfg.flask_port))
    console.print(table)


if __name__ == "__main__":
    app()
