from pathlib import Path

from faster_whisper import WhisperModel


class Transcriber:
    """Transcribes audio files to timestamped text using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        # Model is downloaded on first use to ~/.cache/huggingface/
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path, output_path: Path | None = None) -> str:
        """
        Transcribe an audio file to text.

        Returns the full transcript string. Optionally saves to output_path as .txt.
        Each line format: [HH:MM:SS --> HH:MM:SS] text
        """
        segments, _info = self._model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            language=None,  # auto-detect
        )

        lines = []
        for segment in segments:
            start = _fmt_time(segment.start)
            end = _fmt_time(segment.end)
            lines.append(f"[{start} --> {end}] {segment.text.strip()}")

        transcript = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(transcript, encoding="utf-8")

        return transcript


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
