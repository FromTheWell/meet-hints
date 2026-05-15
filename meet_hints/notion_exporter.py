from datetime import datetime

from notion_client import Client

from .analyzer import MeetingAnalysis

_BLOCK_CHAR_LIMIT = 1800  # Notion paragraph limit is 2000; use 1800 for safety
_BLOCKS_PER_REQUEST = 100  # Notion API limit per append call


class NotionExporter:
    def __init__(self, api_key: str, parent_page_id: str):
        self._client = Client(auth=api_key)
        self._parent_id = parent_page_id

    def create_meeting_page(
        self,
        title: str,
        analysis: MeetingAnalysis,
        transcript: str,
    ) -> str:
        """Create a Notion page with the meeting notes. Returns the page URL."""
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        page_title = f"{date_str} — {title}"

        action_blocks = [_to_do(i) for i in analysis.action_items] or [_paragraph("No action items identified.")]
        decision_blocks = [_bullet(d) for d in analysis.key_decisions] or [_paragraph("No decisions identified.")]
        attendee_blocks = [_bullet(a) for a in analysis.attendees_mentioned] or [_paragraph("No attendees identified.")]

        blocks = [
            _heading("Summary"),
            _paragraph(analysis.summary),
            _divider(),
            _heading("Action Items"),
            *action_blocks,
            _divider(),
            _heading("Key Decisions"),
            *decision_blocks,
            _divider(),
            _heading("Attendees Mentioned"),
            *attendee_blocks,
            _divider(),
            _heading("Full Transcript"),
            *_chunked_paragraphs(transcript),
        ]

        # Create page with first batch of blocks
        first_batch = blocks[:_BLOCKS_PER_REQUEST]
        response = self._client.pages.create(
            parent={"page_id": self._parent_id},
            properties={
                "title": {"title": [{"text": {"content": page_title}}]}
            },
            children=first_batch,
        )
        page_id = response["id"]

        # Append remaining blocks in batches
        remaining = blocks[_BLOCKS_PER_REQUEST:]
        while remaining:
            batch, remaining = remaining[:_BLOCKS_PER_REQUEST], remaining[_BLOCKS_PER_REQUEST:]
            self._client.blocks.children.append(block_id=page_id, children=batch)

        return response["url"]


# --- Block builder helpers ---

def _heading(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": [{"text": {"content": text}}]},
    }


def _paragraph(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"text": {"content": text[:_BLOCK_CHAR_LIMIT]}}]},
    }


def _to_do(text: str) -> dict:
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [{"text": {"content": text[:_BLOCK_CHAR_LIMIT]}}],
            "checked": False,
        },
    }


def _bullet(text: str) -> dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": [{"text": {"content": text[:_BLOCK_CHAR_LIMIT]}}]},
    }


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _chunked_paragraphs(text: str) -> list[dict]:
    return [
        _paragraph(text[i : i + _BLOCK_CHAR_LIMIT])
        for i in range(0, len(text), _BLOCK_CHAR_LIMIT)
    ]
