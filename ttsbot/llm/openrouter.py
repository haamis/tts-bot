import asyncio
import logging
import re
from typing import Any

from openai import (
    APIConnectionError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from ttsbot.parser import Turn

log = logging.getLogger("ttsbot.llm")

# "openrouter/free" is a router that picks from whatever free models are
# currently available, so it never goes stale.
DEFAULT_MODEL = "openrouter/free"

# Routed free models are often reasoning models: they spend hidden reasoning
# tokens before any visible text. A budget sized for the answer alone
# (max_chars // 3) gets fully consumed by thinking -> empty content.
RATE_LIMIT_RETRY_SECONDS = 5.0
MIN_MAX_TOKENS = 1024
REASONING_HEADROOM_TOKENS = 512


class LlmTextTooLong(Exception):
    pass


class LlmRateLimited(Exception):
    pass


class LlmDialogueError(Exception):
    pass


# Fallback !generate prompts when config/generate.yaml is missing or omits a
# key (user overrides merge over these, so the yaml only needs the keys
# being tweaked).
DEFAULT_PROMPTS = {
    "monologue_system": (
        "You write short spoken monologues for a text-to-speech voice bot. "
        "The text is read aloud exactly as written. "
        "Write plain, speakable prose under {max_chars} characters. "
        "Never use markdown: asterisks, underscores and backticks are read "
        "aloud too, so never use them for emphasis — spell it out in words "
        "instead. No stage directions, no lists, no emojis, no sound "
        "effects, no headings. One voice speaking throughout. "
        "Stay in character and be entertaining."
    ),
    "dialogue_system": (
        "You write short spoken dialogues for a text-to-speech voice bot. "
        "There are exactly {num_speakers} speakers: {speaker_names}. "
        "Act out the user's prompt as a natural conversation between them, "
        "taking turns speaking. "
        "Output one turn per line. Every line must start with the "
        "speaker's name followed by a colon, for example:\n"
        "{example_lines}\n"
        "Only the speakers above may speak — never invent other speakers "
        "and never write narration or stage directions. The text is read "
        "aloud exactly as written, so never use markdown: asterisks, "
        "underscores and backticks are read aloud too, never use them for "
        "emphasis — spell it out in words instead. No lists, no emojis, "
        "no headings. Plain, speakable "
        "prose, {max_chars} characters total or fewer. "
        "Stay in character and be entertaining."
    ),
    "shorten_retry": (
        "Too long ({length} characters). Rewrite it under {max_chars} characters."
    ),
}


def load_prompts(path) -> dict:
    """Load config/generate.yaml merged over DEFAULT_PROMPTS.

    Missing file or bad content logs a warning and yields the defaults, so
    !generate never breaks on a config typo (it just ignores it).
    """
    import yaml

    prompts = dict(DEFAULT_PROMPTS)
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key in DEFAULT_PROMPTS:
            if isinstance(data.get(key), str) and data[key].strip():
                prompts[key] = data[key]
    except Exception as e:
        log.warning("Could not load prompts from %s (%s); using defaults", path, e)
    return prompts


class OpenRouterClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: float = 120.0,
                 prompts: dict | None = None):
        self.model = model
        self.prompts = {**DEFAULT_PROMPTS, **(prompts or {})}
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
        )

    def _system_prompt(self, max_chars: int) -> str:
        return self.prompts["monologue_system"].format(max_chars=max_chars)

    def _dialogue_system_prompt(self, voices: list[str], max_chars: int) -> str:
        example = "\n".join(f"{v}: what {v} says" for v in voices)
        return self.prompts["dialogue_system"].format(
            num_speakers=len(voices),
            speaker_names=", ".join(voices),
            example_lines=example,
            max_chars=max_chars,
        )

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        # Some models embed chain-of-thought directly in the content as
        # <think>...</think> (rather than the separate reasoning field);
        # strip it so it never reaches the length check or TTS. An unclosed
        # block means the response was truncated mid-thought — everything
        # after <think> is reasoning, drop it too (may leave empty text,
        # which the empty-retry handles).
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL)
        text = text.strip()
        # Some models wrap the whole monologue in quotes
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        return text

    async def generate(self, scenario: str, max_chars: int) -> str:
        """Single-voice monologue."""
        return await self._request(scenario, max_chars, self._system_prompt(max_chars))

    async def generate_dialogue(
        self, scenario: str, max_chars: int, voices: list[str]
    ) -> list[Turn]:
        """Multi-voice dialogue; the model emits one `voice: text` line per turn."""
        text = await self._request(
            scenario, max_chars, self._dialogue_system_prompt(voices, max_chars)
        )
        return parse_llm_dialogue(text, voices)

    async def _request(self, scenario: str, max_chars: int, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": scenario},
        ]
        max_tokens = max(MIN_MAX_TOKENS, max_chars // 3 + REASONING_HEADROOM_TOKENS)

        text = ""
        # We only need plain prose. "effort: none" disables reasoning
        # entirely (unlike exclude=True, which hides reasoning but still
        # burns output tokens on it) and steers the free router toward
        # non-reasoning models.
        disable_reasoning = True
        # Upstream providers the router landed on and that failed (404/5xx);
        # retried requests exclude them so the router picks another upstream.
        ignored_providers: set[str] = set()
        for attempt in (1, 2, 3):
            request_kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                temperature=0.9,
                max_tokens=max_tokens,
            )
            extra: dict[str, Any] = {}
            if disable_reasoning:
                extra["reasoning"] = {"effort": "none"}
            if ignored_providers:
                extra["provider"] = {"ignore": sorted(ignored_providers)}
            if extra:
                request_kwargs["extra_body"] = extra
            try:
                resp = await self.client.chat.completions.create(**request_kwargs)
            except BadRequestError as e:
                if disable_reasoning and "Reasoning is mandatory" in str(e):
                    # The router can land on an endpoint that cannot disable
                    # reasoning (HTTP 400); retry once without the override —
                    # the model may think, but max_tokens headroom + the
                    # empty-retry below cover it.
                    log.info("Routed endpoint requires reasoning; retrying without reasoning override")
                    disable_reasoning = False
                    continue
                raise
            except RateLimitError:
                if attempt == 1:
                    log.warning("LLM rate-limited (HTTP 429); retrying once in %.0fs", RATE_LIMIT_RETRY_SECONDS)
                    await asyncio.sleep(RATE_LIMIT_RETRY_SECONDS)
                    continue
                raise LlmRateLimited(
                    "LLM is rate-limited right now — try again in a minute"
                ) from None
            except (NotFoundError, InternalServerError, APIConnectionError) as e:
                # The free router can land on a dead upstream endpoint (404
                # surfaced by OpenRouter, e.g. "Provider returned error") or
                # transient 5xx/connection failures. An identical retry can
                # route to the same dead provider again, so the failing
                # provider (when OpenRouter names it) is excluded from the
                # next attempt via the provider.ignore routing option.
                if attempt < 3:
                    provider = self._failed_provider(e)
                    if provider:
                        ignored_providers.add(provider)
                    log.warning(
                        "LLM upstream error (%s: %.120s); retrying%s",
                        type(e).__name__, str(e),
                        f", ignoring provider {provider!r}" if provider else "",
                    )
                    continue
                raise

            if not getattr(resp, "choices", None) or getattr(resp.choices[0], "message", None) is None:
                # OpenRouter can answer HTTP 200 with a null/empty choices
                # list (broken routed upstream) — same as an empty reply:
                # retry routes elsewhere.
                if attempt == 1:
                    log.warning(
                        "LLM response had no usable choices (routed=%s); retrying once",
                        getattr(resp, "model", "?"),
                    )
                    continue
                raise RuntimeError(
                    f"LLM returned no usable choices (routed={getattr(resp, 'model', '?')!r})"
                )

            choice = resp.choices[0]
            text = self._clean(choice.message.content or "")
            if not text:
                # The free router can pick a different model per request, so
                # an empty reply (e.g. reasoning consumed the whole budget,
                # or a content filter) is transient — one retry usually lands
                # a different model.
                if attempt == 1:
                    log.info(
                        "LLM returned empty text (finish_reason=%s, routed=%s); retrying",
                        choice.finish_reason, resp.model,
                    )
                    continue
                raise RuntimeError(
                    "LLM returned empty text "
                    f"(finish_reason={choice.finish_reason!r}, routed={resp.model!r})"
                )
            if len(text) <= max_chars:
                log.info("LLM produced %d chars (attempt %d)", len(text), attempt)
                return text
            if attempt == 1:
                log.info(
                    "LLM output too long (%d > %d), retrying with shorten instruction",
                    len(text), max_chars,
                )
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": self.prompts["shorten_retry"].format(
                        length=len(text), max_chars=max_chars
                    ),
                })
            else:
                # Already rewrote once and it is still too long — give up.
                break

        raise LlmTextTooLong(
            f"LLM produced {len(text)} characters, over the {max_chars} limit"
        )

    @staticmethod
    def _failed_provider(exc: Exception) -> str | None:
        """The upstream provider OpenRouter blames for a 404/5xx, if named.

        Error body shape:
        {"error": {"message": "Provider returned error", "code": 404,
                   "metadata": {"provider_name": "Nvidia", ...}}}
        """
        body = getattr(exc, "body", None)
        if not isinstance(body, dict):
            return None
        try:
            meta = body["error"]["metadata"] or {}
        except (KeyError, TypeError):
            return None
        name = meta.get("provider_name") if isinstance(meta, dict) else None
        return name or None


# After the speaker name: optional markdown emphasis residue, then `name:`
# (space optional) or `name - text` / `name — text` (dashes need surrounding
# spaces so hyphenated words never split into a new turn)
_TURN_LINE_TAIL = r"[*_>\s]*(?::[*_>\s]*|\s+[-\u2013\u2014]\s+)(?P<text>.+)$"


def parse_llm_dialogue(text: str, voices: list[str]) -> list[Turn]:
    """Parse the one-turn-per-line dialogue format into Turns.

    Strictly line-based by design: unlike !speak's %tag parser, a stray % in
    LLM text stays literal and can never reroute voices. Lines without a
    known speaker prefix (preamble chatter, markdown bullets) are dropped;
    if nothing usable remains, LlmDialogueError is raised.
    """
    lowered = {v.lower(): v for v in voices}
    names = "|".join(sorted((re.escape(v) for v in lowered), key=len, reverse=True))
    pattern = re.compile(rf"^(?P<voice>{names}){_TURN_LINE_TAIL}", re.IGNORECASE)

    turns: list[Turn] = []
    for raw in text.splitlines():
        # Tolerate markdown bullets / numbering before the speaker name
        line = re.sub(r"^(?:\d+[.)]|[*_>#\-\s])+", "", raw.strip())
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            log.info("Dropping unprefixed dialogue line: %.80r", line)
            continue
        body = m.group("text").strip()
        # Models like to wrap lines in quotation marks; TTS would read them.
        if len(body) >= 2 and body[0] == body[-1] and body[0] in "\"'":
            body = body[1:-1].strip()
        body = body.strip("*_ ").strip()
        if not body:
            continue
        turns.append(Turn(voice=lowered[m.group("voice").lower()], text=body))

    if not turns:
        raise LlmDialogueError(
            "LLM returned no usable dialogue lines "
            "(expected one 'voice: text' line per turn)"
        )
    return turns