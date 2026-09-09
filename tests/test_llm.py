import httpx
import pytest
from openai import BadRequestError, NotFoundError, RateLimitError

import ttsbot.llm.openrouter as openrouter_mod
from ttsbot.llm.openrouter import (
    OpenRouterClient,
    LlmTextTooLong,
    LlmRateLimited,
    LlmDialogueError,
    parse_llm_dialogue,
)
from ttsbot.parser import Turn


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content, finish_reason="stop", empty=False):
        self.choices = None if empty else [FakeChoice(content, finish_reason)]
        self.model = "some/routed-model:free"


class FakeCompletions:
    """Stands in for client.chat.completions; records calls, replays canned
    responses in order. An Exception item is raised instead of returned;
    None item yields a 200-response with choices=None (broken upstream)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0) if self.responses else ""
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return FakeResponse(*item)
        if item is None:
            return FakeResponse("", empty=True)
        return FakeResponse(item)


def make_client(responses):
    client = OpenRouterClient(api_key="test-key", model="openrouter/free")
    client.client.chat.completions = FakeCompletions(responses)
    return client


def rate_limit_error():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )


def bad_request_error(message):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


def reasoning_mandatory_error():
    return bad_request_error(
        "Reasoning is mandatory for this endpoint and cannot be disabled."
    )


def status_error(cls, code, message):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return cls(message, response=httpx.Response(code, request=request), body=None)


def upstream_404_error():
    return status_error(NotFoundError, 404, "Provider returned error")


def provider_error(cls, code, provider_name):
    """404/5xx with OpenRouter's upstream metadata naming the provider."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    body = {
        "error": {
            "message": "Provider returned error",
            "code": code,
            "metadata": {"raw": "", "provider_name": provider_name, "is_byok": False},
        }
    }
    return cls(
        f"Error code: {code} - {body}",
        response=httpx.Response(code, request=request),
        body=body,
    )


@pytest.mark.asyncio
async def test_generate_happy_path():
    client = make_client(["Hello there, general listener!"])
    text = await client.generate("a greeting", max_chars=1000)
    assert text == "Hello there, general listener!"
    assert len(client.client.chat.completions.calls) == 1
    kwargs = client.client.chat.completions.calls[0]
    assert kwargs["model"] == "openrouter/free"
    assert "1000" in kwargs["messages"][0]["content"]
    # We only want plain prose — reasoning must be disabled
    assert kwargs["extra_body"] == {"reasoning": {"effort": "none"}}


@pytest.mark.asyncio
async def test_generate_strips_wrapping_quotes():
    client = make_client(['"A quoted monologue."'])
    text = await client.generate("topic", max_chars=1000)
    assert text == "A quoted monologue."


@pytest.mark.asyncio
async def test_generate_strips_embedded_think_tags():
    client = make_client(["<think>let me plan the bit</think>The actual monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "The actual monologue."


@pytest.mark.asyncio
async def test_generate_think_tags_not_counted_toward_limit():
    # 900 chars of visible prose + a fat think block must pass the 1000 limit
    prose = "d" * 900
    client = make_client([f"<think>x</think>{prose}"])
    text = await client.generate("topic", max_chars=1000)
    assert text == prose


@pytest.mark.asyncio
async def test_generate_unclosed_think_block_is_reasoning_then_retries():
    # Truncation mid-thought: everything after <think> is reasoning, so the
    # visible text is empty -> the empty-retry lands the next response
    client = make_client([("<think>partial reasoning, no close", "length"), "Recovered monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Recovered monologue."
    assert len(client.client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_generate_overcap_retries_once():
    long = "a" * 1100
    short = "b" * 900
    client = make_client([long, short])
    text = await client.generate("topic", max_chars=1000)
    assert text == short
    assert len(client.client.chat.completions.calls) == 2
    # retry must include the overlong reply + shorten instruction
    messages = client.client.chat.completions.calls[1]["messages"]
    assert messages[-2]["role"] == "assistant"
    assert "Rewrite it under 1000 characters" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_generate_overcap_twice_raises():
    client = make_client(["a" * 1100, "c" * 1200])
    with pytest.raises(LlmTextTooLong):
        await client.generate("topic", max_chars=1000)


@pytest.mark.asyncio
async def test_generate_empty_response_raises():
    client = make_client(["   "])
    with pytest.raises(RuntimeError, match="empty text"):
        await client.generate("topic", max_chars=1000)


@pytest.mark.asyncio
async def test_generate_empty_retries_once():
    client = make_client(["", "Recovered monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Recovered monologue."
    assert len(client.client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_generate_empty_includes_finish_reason_and_routed_model():
    client = make_client([("   ", "length"), ("", "length")])
    with pytest.raises(RuntimeError, match=r"empty text.*'length'.*routed-model"):
        await client.generate("topic", max_chars=1000)


@pytest.mark.asyncio
async def test_generate_max_tokens_leaves_reasoning_headroom():
    client = make_client(["hi"])
    await client.generate("topic", max_chars=1000)
    assert client.client.chat.completions.calls[0]["max_tokens"] >= 1024


@pytest.mark.asyncio
async def test_generate_rate_limited_retries_once(monkeypatch):
    monkeypatch.setattr(openrouter_mod, "RATE_LIMIT_RETRY_SECONDS", 0)
    client = make_client([rate_limit_error(), "Late monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Late monologue."
    assert len(client.client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_generate_rate_limited_twice_raises(monkeypatch):
    monkeypatch.setattr(openrouter_mod, "RATE_LIMIT_RETRY_SECONDS", 0)
    client = make_client([rate_limit_error(), rate_limit_error()])
    with pytest.raises(LlmRateLimited):
        await client.generate("topic", max_chars=1000)


@pytest.mark.asyncio
async def test_generate_reasoning_mandatory_falls_back():
    client = make_client([reasoning_mandatory_error(), "Plain monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Plain monologue."
    calls = client.client.chat.completions.calls
    assert len(calls) == 2
    assert "extra_body" in calls[0]
    assert "extra_body" not in calls[1]


@pytest.mark.asyncio
async def test_generate_other_bad_request_propagates():
    client = make_client([bad_request_error("malformed payload")])
    with pytest.raises(BadRequestError, match="malformed payload"):
        await client.generate("topic", max_chars=1000)
    assert len(client.client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_generate_upstream_404_retries_once():
    """Free-router upstreams can be dead (404); a retry routes elsewhere."""
    client = make_client([upstream_404_error(), "Recovered monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Recovered monologue."
    assert len(client.client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_generate_upstream_404_twice_then_recovers():
    """An identical retry can route to the same dead upstream; a third
    attempt is allowed before giving up."""
    client = make_client([upstream_404_error(), upstream_404_error(), "Recovered monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Recovered monologue."
    assert len(client.client.chat.completions.calls) == 3


@pytest.mark.asyncio
async def test_generate_upstream_404_thrice_propagates():
    client = make_client([upstream_404_error(), upstream_404_error(), upstream_404_error()])
    with pytest.raises(NotFoundError, match="Provider returned error"):
        await client.generate("topic", max_chars=1000)


@pytest.mark.asyncio
async def test_generate_upstream_error_ignores_failed_provider():
    """The failed provider is excluded from the retry via provider.ignore."""
    client = make_client([
        provider_error(NotFoundError, 404, "Nvidia"),
        "Recovered monologue.",
    ])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Recovered monologue."
    kwargs = client.client.chat.completions.calls[1]
    assert kwargs["extra_body"]["provider"]["ignore"] == ["Nvidia"]
    # reasoning override survives alongside the ignore list
    assert kwargs["extra_body"]["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_generate_upstream_error_without_provider_metadata_still_retries():
    client = make_client([upstream_404_error(), "Recovered monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Recovered monologue."
    # No provider metadata -> no ignore list, just the plain retry
    assert "provider" not in client.client.chat.completions.calls[1].get("extra_body", {})


@pytest.mark.asyncio
async def test_generate_null_choices_retries_once():
    """OpenRouter can answer HTTP 200 with choices=None (broken upstream)."""
    client = make_client([None, "Recovered monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "Recovered monologue."
    assert len(client.client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_generate_null_choices_twice_raises():
    client = make_client([None, None])
    with pytest.raises(RuntimeError, match="no usable choices"):
        await client.generate("topic", max_chars=1000)


# --- multi-voice dialogue ----------------------------------------------------


@pytest.mark.asyncio
async def test_generate_dialogue_happy_path():
    client = make_client(["trump: burgers are great\nsnake: no, tacos"])
    turns = await client.generate_dialogue("argue", max_chars=1000, voices=["trump", "snake"])
    assert turns == [Turn(voice="trump", text="burgers are great"), Turn(voice="snake", text="no, tacos")]
    prompt = client.client.chat.completions.calls[0]["messages"][0]["content"]
    assert "trump, snake" in prompt
    assert "one turn per line" in prompt
    assert "trump: what trump says" in prompt


@pytest.mark.asyncio
async def test_generate_dialogue_case_insensitive_prefix():
    turns = parse_llm_dialogue("Trump: hello\nSNAKE: hi there", ["trump", "snake"])
    assert [(t.voice, t.text) for t in turns] == [("trump", "hello"), ("snake", "hi there")]


@pytest.mark.asyncio
async def test_generate_dialogue_tolerates_markdown_and_numbering():
    text = (
        "Here's the dialogue:\n"
        "1. **Trump:** first line\n"
        "* snake: second line\n"
        "> trump: third line"
    )
    turns = parse_llm_dialogue(text, ["trump", "snake"])
    assert [(t.voice, t.text) for t in turns] == [
        ("trump", "first line"),
        ("snake", "second line"),
        ("trump", "third line"),
    ]


@pytest.mark.asyncio
async def test_generate_dialogue_dash_separator():
    turns = parse_llm_dialogue("trump - hello\nsnake — hi", ["trump", "snake"])
    assert [(t.voice, t.text) for t in turns] == [("trump", "hello"), ("snake", "hi")]


@pytest.mark.asyncio
async def test_generate_dialogue_percent_stays_literal():
    """A stray % must never reroute voices (unlike !speak's tag parser)."""
    turns = parse_llm_dialogue("trump: 100% sure %snake loses", ["trump", "snake"])
    assert turns == [Turn(voice="trump", text="100% sure %snake loses")]


@pytest.mark.asyncio
async def test_generate_dialogue_unwraps_line_quotes():
    turns = parse_llm_dialogue('trump: "hello there"\nsnake: \'general...\'', ["trump", "snake"])
    assert [(t.voice, t.text) for t in turns] == [
        ("trump", "hello there"),
        ("snake", "general..."),
    ]


@pytest.mark.asyncio
async def test_generate_dialogue_drops_unprefixed_lines():
    text = "trump: keep this\nsome random chatter without a prefix\nsnake: and this"
    turns = parse_llm_dialogue(text, ["trump", "snake"])
    assert [(t.voice, t.text) for t in turns] == [
        ("trump", "keep this"),
        ("snake", "and this"),
    ]


@pytest.mark.asyncio
async def test_generate_dialogue_prefix_must_touch_separator():
    """'snakes are cool: ...' must not parse as voice 'snake'."""
    with pytest.raises(LlmDialogueError):
        parse_llm_dialogue("snakes are cool: yes", ["snake"])


@pytest.mark.asyncio
async def test_generate_dialogue_nothing_usable_raises():
    with pytest.raises(LlmDialogueError, match="no usable dialogue"):
        parse_llm_dialogue("just some prose, no prefixes at all", ["trump", "snake"])


@pytest.mark.asyncio
async def test_generate_dialogue_empty_text_raises():
    with pytest.raises(LlmDialogueError):
        parse_llm_dialogue("", ["trump", "snake"])


@pytest.mark.asyncio
async def test_generate_dialogue_overcap_retry_still_parses():
    long = "trump: " + "a" * 1100
    short = "trump: short\nsnake: reply"
    client = make_client([long, short])
    turns = await client.generate_dialogue("argue", max_chars=1000, voices=["trump", "snake"])
    assert [(t.voice, t.text) for t in turns] == [("trump", "short"), ("snake", "reply")]
    assert len(client.client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_generate_dialogue_monologue_untouched():
    """generate() (single voice) must keep the monologue system prompt."""
    client = make_client(["A monologue."])
    text = await client.generate("topic", max_chars=1000)
    assert text == "A monologue."
    prompt = client.client.chat.completions.calls[0]["messages"][0]["content"]
    assert "monologues" in prompt