"""Save a call as a full conversation or a meeting resume.

The resume only restates lines that were captured. It does not add
agreements, dates, or decisions that nobody said.
"""

import csv
import io
import os
import re
from typing import Iterable, List, Sequence, Tuple

Turn = Tuple[str, str, str]
Suggestion = Tuple[str, str, str]

FORMATS = (
    ("md", "Markdown", ".md"),
    ("txt", "Text", ".txt"),
    ("csv", "CSV", ".csv"),
)
KIND_CONVERSATION = "conversation"
KIND_RESUME = "resume"

_TURN_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]\s*([^:\n]+):\s*(.+)")


def turns_from_packet(markdown: str) -> List[Turn]:
    """Pull captured lines out of a saved review packet, in spoken order."""
    seen = set()
    turns: List[Turn] = []
    for match in _TURN_RE.finditer(markdown or ""):
        turn = (match.group(1), match.group(2).strip(), match.group(3).strip())
        if turn in seen or not turn[2]:
            continue
        seen.add(turn)
        turns.append(turn)
    return turns


def profile_from_packet(markdown: str, fallback: str = "") -> str:
    match = re.search(r"(?:Target Profile|Workspace Profile)\**:?\s*([^\n|]+)", markdown or "")
    if not match:
        return fallback
    return match.group(1).strip()


def render_document(
    kind: str,
    fmt: str,
    turns: Sequence[Turn],
    suggestions: Sequence[Suggestion] = (),
    profile: str = "",
    recorded: str = "",
) -> str:
    """Render a conversation or a meeting resume in md, txt, or csv."""
    if fmt == "csv":
        return _as_csv(kind, turns, suggestions, profile, recorded)
    if fmt == "txt":
        return _as_text(kind, turns, suggestions, profile, recorded)
    return _as_markdown(kind, turns, suggestions, profile, recorded)


def default_filename(kind: str, fmt: str, stamp: str) -> str:
    stem = "meeting_resume" if kind == KIND_RESUME else "conversation"
    suffix = next((item[2] for item in FORMATS if item[0] == fmt), ".md")
    return f"{stem}_{stamp}{suffix}"


def ensure_extension(path: str, fmt: str) -> str:
    root, ext = os.path.splitext(path)
    known = {item[2] for item in FORMATS}
    if ext.lower() in known:
        return path
    suffix = next((item[2] for item in FORMATS if item[0] == fmt), ".md")
    return path + suffix


def _as_markdown(kind, turns, suggestions, profile, recorded) -> str:
    title = "Meeting resume" if kind == KIND_RESUME else "Conversation"
    lines = [f"# {title}", ""]
    lines.extend(_header_lines(profile, recorded, markdown=True))
    if kind == KIND_RESUME:
        lines.extend(_overview_lines(turns, markdown=True))
        lines.extend(["", "## Minutes", ""])
    else:
        lines.extend(["", "## Dialogue", ""])
    if not turns:
        lines.append("_No speech was captured._")
    for stamp, speaker, text in turns:
        lines.append(f"- [{stamp}] **{speaker}**: {text}")
    if kind == KIND_RESUME:
        lines.extend(["", "## Suggested replies", ""])
        if suggestions:
            lines.append("These are replies the copilot suggested. They are not agreements from the call.")
            lines.append("")
            for stamp, speaker, text in suggestions:
                lines.append(f"- [{stamp}] After {speaker}: {text}")
        else:
            lines.append("_No suggested replies were generated._")
    lines.append("")
    return "\n".join(lines)


def _as_text(kind, turns, suggestions, profile, recorded) -> str:
    title = "Meeting resume" if kind == KIND_RESUME else "Conversation"
    lines = [title, ""]
    lines.extend(_header_lines(profile, recorded, markdown=False))
    if kind == KIND_RESUME:
        lines.extend(_overview_lines(turns, markdown=False))
        lines.extend(["", "Minutes", ""])
    else:
        lines.extend(["", "Dialogue", ""])
    if not turns:
        lines.append("No speech was captured.")
    for stamp, speaker, text in turns:
        lines.append(f"[{stamp}] {speaker}: {text}")
    if kind == KIND_RESUME:
        lines.extend(["", "Suggested replies", ""])
        if suggestions:
            lines.append("These are replies the copilot suggested. They are not agreements from the call.")
            lines.append("")
            for stamp, speaker, text in suggestions:
                lines.append(f"[{stamp}] After {speaker}: {text}")
        else:
            lines.append("No suggested replies were generated.")
    lines.append("")
    return "\n".join(lines)


def _as_csv(kind, turns, suggestions, profile, recorded) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if kind == KIND_RESUME:
        writer.writerow(["section", "time", "speaker", "text"])
        writer.writerow(["overview", "", "", f"Profile: {profile or 'Unknown'}"])
        writer.writerow(["overview", "", "", f"Recorded: {recorded}"])
        for label, count in _speaker_counts(turns):
            writer.writerow(["overview", "", label, f"{count} turns"])
        for stamp, speaker, text in turns:
            writer.writerow(["minute", stamp, speaker, text])
        for stamp, speaker, text in suggestions:
            writer.writerow(["suggestion", stamp, speaker, text])
    else:
        writer.writerow(["time", "speaker", "text"])
        for stamp, speaker, text in turns:
            writer.writerow([stamp, speaker, text])
    return buffer.getvalue()


def _header_lines(profile: str, recorded: str, markdown: bool) -> Iterable[str]:
    profile_label = profile or "Unknown"
    if markdown:
        return [f"**Profile**: {profile_label}", f"**Recorded**: {recorded}"]
    return [f"Profile: {profile_label}", f"Recorded: {recorded}"]


def _overview_lines(turns: Sequence[Turn], markdown: bool) -> List[str]:
    heading = "## Overview" if markdown else "Overview"
    lines = [
        "",
        heading,
        "",
        "This resume restates the captured transcript. It does not add agreements, dates, or decisions that were not spoken.",
        "",
    ]
    if not turns:
        lines.append("No speech was captured.")
        return lines
    lines.append(f"First line: {turns[0][0]}")
    lines.append(f"Last line: {turns[-1][0]}")
    lines.append(f"Turns: {len(turns)}")
    for speaker, count in _speaker_counts(turns):
        lines.append(f"{speaker}: {count} turns")
    return lines


def _speaker_counts(turns: Sequence[Turn]) -> List[Tuple[str, int]]:
    counts = {}
    order = []
    for _stamp, speaker, _text in turns:
        if speaker not in counts:
            order.append(speaker)
            counts[speaker] = 0
        counts[speaker] += 1
    return [(speaker, counts[speaker]) for speaker in order]
