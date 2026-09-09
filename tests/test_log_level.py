import logging

from ttsbot.bot import apply_log_level


def test_apply_log_level_valid():
    apply_log_level("warning")
    assert logging.getLogger().level == logging.WARNING
    apply_log_level("debug")
    assert logging.getLogger().level == logging.DEBUG


def test_apply_log_level_invalid_falls_back_to_info():
    apply_log_level("bogus")
    assert logging.getLogger().level == logging.INFO
