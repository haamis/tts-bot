from ttsbot.bot import TTSBot

extract = TTSBot._extract_url


def test_plain_url():
    assert extract("https://www.youtube.com/watch?v=abc") == "https://www.youtube.com/watch?v=abc"


def test_url_with_surrounding_command_text():
    """The reported failure: the whole command string pasted as the URL."""
    text = "!rvc snake https://www.youtube.com/watch?v=7gCo8fajC7o"
    assert extract(text) == "https://www.youtube.com/watch?v=7gCo8fajC7o"


def test_angle_bracket_url():
    assert extract("<https://youtu.be/abc>") == "https://youtu.be/abc"


def test_trailing_punctuation_stripped():
    assert extract("check https://youtu.be/abc.") == "https://youtu.be/abc"


def test_paren_wrapped_url():
    assert extract("(https://youtu.be/abc)") == "https://youtu.be/abc"
    assert extract("see (https://youtu.be/abc).") == "https://youtu.be/abc"


def test_http_allowed():
    assert extract("http://example.com/a.wav") == "http://example.com/a.wav"


def test_no_url_returns_none():
    assert extract("no url here") is None
    assert extract("") is None
    assert extract(None) is None


def test_scheme_without_netloc_rejected():
    assert extract("https://") is None


def test_first_url_wins():
    text = "!rvc snake https://a.example.com/1 https://b.example.com/2"
    assert extract(text) == "https://a.example.com/1"