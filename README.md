# meet-hints

CLI tool that records meetings, transcribes audio with Whisper, analyzes the transcript with Claude, and exports a structured summary to Notion.

```
Audio Recording → Transcription → Claude Analysis → Notion Export
```

---

## Requirements

- Python ≥ 3.11
- Windows (uses WASAPI loopback for system audio capture)
- Optional: CUDA-capable GPU for faster transcription

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd meet-hints

# (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install globally (registers the meet-hints command)
pip install -e .

# Verify installation
meet-hints --help
```

---

## Configuration

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude API key. If unset, analysis is skipped. |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model to use for analysis. |
| `NOTION_API_KEY` | — | Notion integration token. If unset, export is skipped. |
| `NOTION_PARENT_PAGE_ID` | — | UUID of the Notion page where meeting pages will be created. |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2`. |
| `WHISPER_DEVICE` | `cpu` | Compute device for Whisper: `cpu` or `cuda`. |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization type: `int8`, `int16`, `float16`, `float32`. |
| `OUTPUT_DIR` | `./outputs` | Directory where audio files and transcripts are saved. |
| `AUDIO_SAMPLE_RATE` | `16000` | Sample rate in Hz for recordings. |

**Minimal `.env` for full pipeline:**

```env
ANTHROPIC_API_KEY=sk-ant-...
NOTION_API_KEY=secret_...
NOTION_PARENT_PAGE_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

The tool degrades gracefully: without an Anthropic key it saves the transcript; without Notion credentials it prints the summary to the console.

---

## Commands

### `meet-hints list-devices`

Lists available audio devices. Run this first to find the correct device indices for your setup.

```bash
meet-hints list-devices
```

**Example output:**

```
Loopback devices (use with --device):
  [0] Stereo Mix — 2ch 48000Hz
  [1] CABLE Output — 2ch 44100Hz

Microphone devices (use with --mic):
  [0] Default Microphone — 1ch 16000Hz
  [1] Headset Microphone — 1ch 48000Hz
```

---

### `meet-hints record`

Records system audio (and optionally the microphone) until you press `Ctrl+C`, then runs the full pipeline.

```bash
meet-hints record [OPTIONS]
```

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--title` | `-t` | string | datetime | Title for the Notion page and output files. |
| `--device` | `-d` | int | auto | Loopback device index (from `list-devices`). |
| `--mic` | `-m` | int | none | Microphone device index (from `list-devices`). |
| `--whisper-model` | | string | env value | Override `WHISPER_MODEL_SIZE` for this run. |
| `--no-notion` | | flag | false | Skip Notion export; print summary to console. |

**Examples:**

```bash
# Record system audio only (auto-detect loopback device)
meet-hints record

# Record with a custom title
meet-hints record --title "Q1 Planning"

# Record system audio from device 1 and mix in microphone from device 0
meet-hints record --device 1 --mic 0

# Microphone only (no system audio)
meet-hints record --mic 0

# Use a larger Whisper model for better accuracy
meet-hints record --title "Weekly Sync" --whisper-model large-v2

# Record without exporting to Notion
meet-hints record --no-notion
```

**Audio capture modes:**

| Flags used | Behaviour |
|---|---|
| Neither `--device` nor `--mic` | Auto-detect loopback device |
| `--device` only | Loopback (system audio) |
| `--mic` only | Microphone only |
| `--device` + `--mic` | Mix both streams |

---

### `meet-hints process <audio_file>`

Runs the transcription → analysis → export pipeline on an existing WAV file.

```bash
meet-hints process AUDIO_FILE [OPTIONS]
```

| Argument / Option | Type | Default | Description |
|---|---|---|---|
| `AUDIO_FILE` | path | required | Path to the `.wav` file to process. |
| `--title`, `-t` | string | datetime | Title for the Notion page. |
| `--whisper-model` | string | env value | Override Whisper model for this run. |
| `--no-notion` | flag | false | Skip Notion export. |

**Examples:**

```bash
# Process a recording and export to Notion
meet-hints process outputs/audio/meeting_20260515_143022.wav

# Add a custom title
meet-hints process recording.wav --title "Monday Standup"

# Process without Notion, just print the summary
meet-hints process recording.wav --no-notion

# Use a higher-quality model for transcription
meet-hints process recording.wav --whisper-model medium
```

---

## Pipeline Details

### Transcription

- Uses [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper); model files are downloaded automatically to `~/.cache/huggingface/` on first use.
- Output is a timestamped transcript: `[HH:MM:SS --> HH:MM:SS] text`.
- Language is auto-detected.

### Analysis (Claude)

- Transcripts up to ~150K tokens are processed in a single pass.
- Longer transcripts are split into chunks and analyzed in parallel, then synthesized.
- Extracted structured data:
  - **Summary** — 2–4 sentence overview
  - **Action Items** — verb-based task list
  - **Key Decisions** — outcomes and agreements reached
  - **Attendees** — names and roles mentioned

### Notion Export

- Creates a page under `NOTION_PARENT_PAGE_ID` titled `YYYY-MM-DD HH:MM — {title}`.
- Page contains all structured sections as headings, bullet lists, and to-do items.

---

## Output Files

All files are saved under `OUTPUT_DIR` (default `./outputs`):

```
outputs/
├── audio/
│   └── meeting_YYYYMMDD_HHMMSS.wav
└── transcripts/
    └── meeting_YYYYMMDD_HHMMSS.txt
```

---

## Whisper Model Comparison

| Model | Size | Speed | Accuracy | Recommended for |
|---|---|---|---|---|
| `tiny` | ~75 MB | fastest | lowest | Quick drafts, testing |
| `base` | ~145 MB | fast | good | Default; most meetings |
| `small` | ~465 MB | moderate | better | Technical discussions |
| `medium` | ~1.5 GB | slow | high | Important recordings |
| `large-v2` | ~3 GB | slowest | best | Maximum accuracy |

Use `WHISPER_DEVICE=cuda` and a matching `WHISPER_COMPUTE_TYPE` (`float16`) to run larger models at acceptable speed on a GPU.

---

## Graceful Degradation

| Missing credential | Behaviour |
|---|---|
| No `ANTHROPIC_API_KEY` | Transcript is saved; analysis and Notion export are skipped. |
| No `NOTION_API_KEY` or `NOTION_PARENT_PAGE_ID` | Analysis runs; summary is printed to console instead of exported. |
| `--no-notion` flag | Same as above, regardless of credentials. |
| No loopback devices found | Falls back to microphone-only mode if `--mic` is provided. |
| Empty transcript | Pipeline stops with a warning after transcription. |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pyaudiowpatch` | WASAPI loopback + mic audio capture |
| `faster-whisper` | Speech-to-text transcription |
| `anthropic` | Claude API client |
| `notion-client` | Notion API integration |
| `click` | CLI framework |
| `pydantic-settings` | `.env` configuration loading |
| `rich` | Terminal progress and formatting |
| `numpy` | Audio resampling and mixing |
