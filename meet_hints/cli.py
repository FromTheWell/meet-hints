from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

console = Console()


def _make_transcript_path(audio_path: Path, output_dir: str) -> Path:
    return Path(output_dir).resolve() / "transcripts" / (audio_path.stem + ".txt")


@click.group()
def cli():
    """meet-hints: Record Teams meetings → transcript → Claude summary → Notion."""
    pass


@cli.command()
@click.option("--title", "-t", default=None, help="Meeting title for the Notion page")
@click.option("--no-notion", is_flag=True, help="Skip Notion export (transcript + summary only)")
@click.option("--whisper-model", default=None, help="Whisper model size (tiny/base/small/medium/large-v2)")
@click.option("--device", "-d", default=None, type=int, help="Loopback device index (see list-devices)")
@click.option("--mic", "-m", default=None, type=int, help="Microphone device index (see list-devices)")
def record(title: str | None, no_notion: bool, whisper_model: str | None, device: int | None, mic: int | None):
    """Record system audio. Press Enter to stop and process."""
    from .config import settings
    from .recorder import AudioRecorder

    # Only mic specified → mic-only mode (no loopback)
    mic_only = mic is not None and device is None

    recorder = AudioRecorder(
        output_dir=settings.output_dir,
        sample_rate=settings.audio_sample_rate,
        device_index=device,
        mic_device_index=mic,
        mic_only=mic_only,
    )

    audio_path = recorder.start()

    import time
    time.sleep(0.3)  # give threads time to open streams

    start_time = time.time()

    def _elapsed() -> Text:
        elapsed = int(time.time() - start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        timestamp = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        return Text.from_markup(
            f"[bold red]● REC[/]  [bold]{timestamp}[/]  — press [bold]Ctrl+C[/] to stop."
        )

    try:
        with Live(_elapsed(), refresh_per_second=2, console=console) as live:
            while True:
                time.sleep(0.5)
                live.update(_elapsed())
    except KeyboardInterrupt:
        pass

    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    console.print("[yellow]Stopping...[/]")
    audio_path = recorder.stop()
    console.print(f"[green]Audio saved:[/] {audio_path}")

    _run_pipeline(
        audio_path=audio_path,
        title=title,
        no_notion=no_notion,
        whisper_model=whisper_model,
    )

    signal.signal(signal.SIGINT, signal.SIG_DFL)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--title", "-t", default=None, help="Meeting title for the Notion page")
@click.option("--no-notion", is_flag=True, help="Skip Notion export")
@click.option("--whisper-model", default=None, help="Whisper model size")
def process(audio_file: Path, title: str | None, no_notion: bool, whisper_model: str | None):
    """Process an existing audio file through transcription, analysis, and Notion export."""
    _run_pipeline(
        audio_path=audio_file,
        title=title,
        no_notion=no_notion,
        whisper_model=whisper_model,
    )


@cli.command("list-devices")
def list_devices():
    """List available loopback and microphone devices."""
    from .recorder import AudioRecorder

    recorder = AudioRecorder(output_dir="./outputs")

    loopbacks = recorder.list_loopback_devices()
    mics = recorder.list_input_devices()

    console.print("[bold]Loopback devices[/] (use with --device):")
    if loopbacks:
        for dev in loopbacks:
            console.print(f"  [{int(dev['index'])}] {dev['name']} — {int(dev['defaultSampleRate'])}Hz")
    else:
        console.print("  [red]None found[/]")

    console.print()
    console.print("[bold]Microphone devices[/] (use with --mic):")
    if mics:
        for dev in mics:
            console.print(f"  [{int(dev['index'])}] {dev['name']} — {int(dev['defaultSampleRate'])}Hz")
    else:
        console.print("  [red]None found[/]")


def _run_pipeline(
    audio_path: Path,
    title: str | None,
    no_notion: bool,
    whisper_model: str | None,
):
    from .config import settings
    from .transcriber import Transcriber
    from .analyzer import MeetingAnalyzer
    from .notion_exporter import NotionExporter

    transcript_path = _make_transcript_path(audio_path, settings.output_dir)

    # Step 1: Transcribe
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        progress.add_task("Transcribing audio (this may take a while)...", total=None)
        model_size = whisper_model or settings.whisper_model_size
        transcriber = Transcriber(
            model_size=model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        transcript = transcriber.transcribe(audio_path, output_path=transcript_path)

    if not transcript.strip():
        console.print("[yellow]Warning: transcript is empty — no speech detected.[/]")
        return

    console.print(f"[green]Transcript saved:[/] {transcript_path}")

    # Step 2: Analyze with Claude
    if not settings.anthropic_api_key:
        console.print("[yellow]ANTHROPIC_API_KEY not set — skipping Claude analysis.[/]")
        console.print(f"[dim]Transcript saved to: {transcript_path}[/]")
        return

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        progress.add_task("Analyzing with Claude...", total=None)
        analyzer = MeetingAnalyzer(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )
        analysis = analyzer.analyze(transcript)

    console.print(Panel(
        f"[bold]Summary[/]\n{analysis.summary}\n\n"
        f"[bold]Action Items[/]\n" + "\n".join(f"• {a}" for a in analysis.action_items) + "\n\n"
        f"[bold]Key Decisions[/]\n" + "\n".join(f"• {d}" for d in analysis.key_decisions),
        title="Meeting Analysis",
        border_style="green",
    ))

    # Step 3: Export to Notion
    if no_notion:
        return

    if not settings.notion_api_key or not settings.notion_parent_page_id:
        console.print("[yellow]NOTION_API_KEY or NOTION_PARENT_PAGE_ID not set — skipping Notion export.[/]")
        return

    meeting_title = title or f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        progress.add_task("Creating Notion page...", total=None)
        exporter = NotionExporter(
            api_key=settings.notion_api_key,
            parent_page_id=settings.notion_parent_page_id,
        )
        page_url = exporter.create_meeting_page(
            title=meeting_title,
            analysis=analysis,
            transcript=transcript,
        )

    console.print(f"[bold green]Notion page created:[/] {page_url}")


def main():
    cli()


if __name__ == "__main__":
    main()
