import pytest
from ttsbot.parser import parse_dialogue, ParseError, Turn
from ttsbot.config import Config, VoiceConfig


def test_parse_simple_dialogue():
    known = {"mario", "luigi"}
    text = "%mario hello world %luigi how are you"
    turns = parse_dialogue(text, known, 500)
    assert len(turns) == 2
    assert turns[0] == Turn(voice="mario", text="hello world")
    assert turns[1] == Turn(voice="luigi", text="how are you")


def test_parse_multiple_words_per_turn():
    known = {"snake"}
    text = "%snake this is a longer sentence with multiple words"
    turns = parse_dialogue(text, known, 500)
    assert len(turns) == 1
    assert turns[0].voice == "snake"
    assert turns[0].text == "this is a longer sentence with multiple words"


def test_parse_empty_text_raises():
    known = {"mario"}
    with pytest.raises(ParseError, match="Empty text"):
        parse_dialogue("%mario   %luigi hello", known, 500)


def test_parse_unknown_voice_raises():
    known = {"mario"}
    with pytest.raises(ParseError, match="Unknown voice"):
        parse_dialogue("%luigi hello", known, 500)


def test_parse_no_tags_raises():
    known = {"mario"}
    with pytest.raises(ParseError, match="No voice tags"):
        parse_dialogue("hello world", known, 500)


def test_parse_char_limit():
    known = {"mario"}
    long_text = "%mario " + "a" * 501
    with pytest.raises(ParseError, match="exceeds 500"):
        parse_dialogue(long_text, known, 500)


def test_parse_empty_input_raises():
    known = {"mario"}
    with pytest.raises(ParseError, match="Empty dialogue"):
        parse_dialogue("", known, 500)
    with pytest.raises(ParseError, match="Empty dialogue"):
        parse_dialogue("   ", known, 500)


def test_parse_voice_with_underscore_and_numbers():
    known = {"voice_1", "voice-2"}
    text = "%voice_1 hello %voice-2 world"
    turns = parse_dialogue(text, known, 500)
    assert len(turns) == 2
    assert turns[0].voice == "voice_1"
    assert turns[1].voice == "voice-2"


def test_config_load():
    config = Config.load("config/voices.yaml")
    assert config.default_tts == "en_US-lessac-medium"
    assert config.max_chars == 500
    assert "snake" in config.voices
    assert "trump" in config.voices

    snake = config.voices["snake"]
    assert snake.tts == "en_US-lessac-medium"
    assert snake.rvc_model == "/home/haama/RVC/SSNAKE/SSNAKE.pth"
    assert "IVF967_Flat_SSNAKE" in snake.rvc_index
    assert snake.pitch == 0
    assert snake.index_rate == 0.75
    assert snake.f0_method == "rmvpe"

    trump = config.voices["trump"]
    assert trump.tts == "en_US-ryan-medium"
    assert trump.rvc_model == "/home/haama/RVC/trump/trump.pth"
    assert "IVF1170_Flat_trump" in trump.rvc_index