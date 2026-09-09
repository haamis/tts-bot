import re
from dataclasses import dataclass
from typing import List


@dataclass
class Turn:
    voice: str
    text: str


class ParseError(Exception):
    pass


TAG_PATTERN = re.compile(r"%(?P<name>[a-zA-Z][a-zA-Z0-9_-]*)")


def parse_dialogue(text: str, known_voices: set[str], max_chars: int = 500) -> List[Turn]:
    if not text.strip():
        raise ParseError("Empty dialogue")

    total_chars = len(text)
    if total_chars > max_chars:
        raise ParseError(f"Dialogue exceeds {max_chars} character limit ({total_chars} chars)")

    matches = list(TAG_PATTERN.finditer(text))
    if not matches:
        raise ParseError("No voice tags found. Use %voice_name text format.")

    leading = text[: matches[0].start()].strip()
    if leading:
        raise ParseError(
            f"Text before the first voice tag has no voice: '{leading[:50]}'. "
            "Start with %voice_name."
        )

    turns = []
    for i, match in enumerate(matches):
        voice_name = match.group("name")
        if voice_name not in known_voices:
            raise ParseError(f"Unknown voice: '{voice_name}'. Known voices: {', '.join(sorted(known_voices))}")

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end].strip()

        if not segment:
            raise ParseError(f"Empty text for voice '{voice_name}'")

        turns.append(Turn(voice=voice_name, text=segment))

    return turns