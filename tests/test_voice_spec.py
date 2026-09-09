"""!generate voice spec parsing: 'a,b,c' -> validated, deduped list."""
from ttsbot.bot import TTSBot


class FakeConfig:
    def __init__(self, names):
        self.voices = {n: object() for n in names}

    def get_voice(self, name):
        return self.voices.get(name)


def resolve(known, spec):
    stub = type("S", (), {"config": FakeConfig(known)})()
    return TTSBot._resolve_voice_spec(stub, spec)


def test_multiple_voices_kept_in_order():
    assert resolve(["snake", "trump"], "trump,snake") == ["trump", "snake"]


def test_single_voice_is_monologue_list():
    assert resolve(["snake", "trump"], "trump") == ["trump"]


def test_percent_prefix_stripped():
    assert resolve(["snake"], "%snake") == ["snake"]


def test_duplicates_dropped():
    assert resolve(["snake", "trump"], "trump,trump,snake") == ["trump", "snake"]


def test_whitespace_tolerated():
    assert resolve(["snake", "trump"], " trump , snake ") == ["trump", "snake"]


def test_empty_tokens_skipped():
    assert resolve(["snake"], ",snake,") == ["snake"]


def test_unknown_voice_returns_none():
    assert resolve(["snake"], "trump,snake") is None
    assert resolve(["snake"], "nope") is None


def test_empty_spec_returns_none():
    assert resolve(["snake"], "") is None
    assert resolve(["snake"], ",,") is None
