import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic


@dataclass
class MeetingAnalysis:
    summary: str
    action_items: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    attendees_mentioned: list[str] = field(default_factory=list)


_SYSTEM_PROMPT = (
    "You are an expert meeting analyst. You receive timestamped meeting transcripts "
    "and extract structured information. Always respond with valid JSON matching the "
    "exact schema requested. Be concise but complete."
)

_ANALYSIS_PROMPT = """Analyze this meeting transcript and return a JSON object with exactly these fields:

{{
  "summary": "2-4 sentence summary of the meeting",
  "action_items": ["List of specific action items, each starting with a verb and assignee if mentioned"],
  "key_decisions": ["List of decisions made during the meeting"],
  "attendees_mentioned": ["Names or roles of people mentioned in the transcript"]
}}

TRANSCRIPT:
{transcript}

Respond ONLY with the JSON object, no markdown fences."""

_SYNTHESIS_PROMPT = """Synthesize these section summaries from a long meeting into one coherent 3-4 sentence summary:

{summaries}

Respond with plain text only."""

# ~150K tokens ≈ 600K characters
_MAX_CHARS_SINGLE = 600_000
_CHUNK_LINES = 2000
_CHUNK_OVERLAP = 10


class MeetingAnalyzer:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def analyze(self, transcript: str) -> MeetingAnalysis:
        if len(transcript) <= _MAX_CHARS_SINGLE:
            return self._analyze_single(transcript)
        return self._analyze_chunked(transcript)

    def _analyze_single(self, transcript: str) -> MeetingAnalysis:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": _ANALYSIS_PROMPT.format(transcript=transcript),
                }
            ],
        )
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return MeetingAnalysis(
            summary=data.get("summary", ""),
            action_items=data.get("action_items", []),
            key_decisions=data.get("key_decisions", []),
            attendees_mentioned=data.get("attendees_mentioned", []),
        )

    def _analyze_chunked(self, transcript: str) -> MeetingAnalysis:
        lines = transcript.splitlines()
        chunks: list[str] = []
        i = 0
        while i < len(lines):
            chunk = "\n".join(lines[i : i + _CHUNK_LINES])
            chunks.append(chunk)
            i += _CHUNK_LINES - _CHUNK_OVERLAP

        with ThreadPoolExecutor(max_workers=4) as executor:
            partial_results = list(executor.map(self._analyze_single, chunks))

        # Merge and deduplicate lists
        action_items = list({item for r in partial_results for item in r.action_items})
        key_decisions = list({d for r in partial_results for d in r.key_decisions})
        attendees = list({a for r in partial_results for a in r.attendees_mentioned})

        # Synthesize summaries into one
        combined = "\n\n".join(f"Section {i+1}: {r.summary}" for i, r in enumerate(partial_results))
        synthesis = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": _SYNTHESIS_PROMPT.format(summaries=combined),
                }
            ],
        )
        final_summary = next(b.text for b in synthesis.content if b.type == "text").strip()

        return MeetingAnalysis(
            summary=final_summary,
            action_items=action_items,
            key_decisions=key_decisions,
            attendees_mentioned=attendees,
        )
