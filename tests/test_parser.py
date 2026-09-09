import pytest
from pathlib import Path
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


def test_parse_repeated_voice():
    known = {"mario"}
    turns = parse_dialogue("%mario one %mario two", known, 500)
    assert len(turns) == 2
    assert turns[0].text == "one"
    assert turns[1].text == "two"


def test_parse_percent_in_text_is_not_a_tag():
    known = {"mario"}
    turns = parse_dialogue("%mario 100% sure", known, 500)
    assert turns[0].text == "100% sure"


def test_parse_leading_untagged_text_raises():
    known = {"mario"}
    with pytest.raises(ParseError, match="no voice"):
        parse_dialogue("untagged text %mario hello", known, 500)


def test_config_load():
    config = Config.load("config/voices.yaml")
    assert config.max_chars > 0
    # snake/trump are the documented dry-run voices
    assert "snake" in config.voices
    assert "trump" in config.voices

    for name, voice in config.voices.items():
        # every voice must point at real assets — catches path drift
        model = Path(voice.rvc_model)
        assert model.is_file(), f"{name}: rvc_model not found: {model}"
        if voice.rvc_index:
            index = Path(voice.rvc_index)
            assert index.is_file(), f"{name}: rvc_index not found: {index}"
        assert voice.tts, f"{name}: missing piper voice"
        assert voice.f0_method in ("pm", "rmvpe")
        assert isinstance(voice.speaker_id, int) and voice.speaker_id >= 0
        assert 0 <= voice.index_rate <= 1
        assert voice.speed_cloud > 0
        assert voice.speed_local > 0