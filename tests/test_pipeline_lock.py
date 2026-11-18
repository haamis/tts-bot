"""The per-channel queue: a depth-1 processing pipeline and a depth-1 play slot."""
import asyncio

import pytest

from ttsbot.pipeline import Pipeline


async def _enter(pipeline, request):
    async with pipeline.processing_slot(request):
        return True


@pytest.mark.asyncio
async def test_processing_slot_is_depth_one():
    """A second request waits for the first to finish preparing."""
    p = Pipeline(tts=None, rvc=None)
    order = []

    async def work(name):
        async with p.processing_slot(p.new_request(1, 2)):
            order.append(f"{name}-start")
            await asyncio.sleep(0.01)
            order.append(f"{name}-end")

    await asyncio.gather(work("a"), work("b"))
    assert order == ["a-start", "a-end", "b-start", "b-end"]


@pytest.mark.asyncio
async def test_processing_overlaps_playback():
    """The point of the split: preparing runs while another track plays."""
    p = Pipeline(tts=None, rvc=None)
    order = []

    async def do_play():
        async with p.play_slot(p.new_request(1, 2)):
            order.append("play-start")
            await asyncio.sleep(0.05)
            order.append("play-end")

    async def do_prepare():
        async with p.processing_slot(p.new_request(1, 2)):
            order.append("prepare-start")
            await asyncio.sleep(0.01)
            order.append("prepare-end")

    await asyncio.gather(do_play(), do_prepare())
    assert order.index("prepare-start") < order.index("play-end")
    assert order.index("prepare-end") < order.index("play-end")


@pytest.mark.asyncio
async def test_play_slot_serializes_playback():
    """Only one track plays at a time, even when both are ready."""
    p = Pipeline(tts=None, rvc=None)
    order = []

    async def play(name):
        async with p.play_slot(p.new_request(1, 2)):
            order.append(f"{name}-start")
            await asyncio.sleep(0.01)
            order.append(f"{name}-end")

    await asyncio.gather(play("a"), play("b"))
    assert order == ["a-start", "a-end", "b-start", "b-end"]


@pytest.mark.asyncio
async def test_requests_are_never_rejected():
    """Three requests, one pipeline: all run, in order."""
    p = Pipeline(tts=None, rvc=None)
    order = []

    async def work(name):
        async with p.processing_slot(p.new_request(1, 2)):
            order.append(name)

    await asyncio.gather(work("a"), work("b"), work("c"))
    assert order == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_tracks_ahead_counts_the_whole_backlog():
    p = Pipeline(tts=None, rvc=None)
    assert p.tracks_ahead(1, 2) == 0
    first = p.new_request(1, 2)
    async with p.processing_slot(first):
        second = asyncio.create_task(_enter(p, p.new_request(1, 2)))
        await asyncio.sleep(0)  # let it queue on the slot
        # The request in the pipeline plus the one queued behind it.
        assert p.tracks_ahead(1, 2) == 2
    await asyncio.wait_for(second, timeout=5)
    assert p.tracks_ahead(1, 2) == 0


@pytest.mark.asyncio
async def test_tracks_ahead_is_per_channel():
    p = Pipeline(tts=None, rvc=None)
    async with p.processing_slot(p.new_request(1, 2)):
        assert p.tracks_ahead(1, 2) == 1
        assert p.tracks_ahead(1, 3) == 0


@pytest.mark.asyncio
async def test_prepared_request_waits_for_the_play_slot():
    """A prepared track is queued for playback, not for processing."""
    p = Pipeline(tts=None, rvc=None)
    request = p.new_request(1, 2)
    p.mark_prepared(request)
    assert p.tracks_ahead(1, 2) == 1
    assert p.cancel_target(1, 2) is request
    async with p.play_slot(request):
        # While it plays it is the playing request.
        assert p.cancel_target(1, 2) is request
    # Leaving the slot clears it from the prepared list: nothing left to cancel.
    assert p.cancel_target(1, 2) is None
    assert p.tracks_ahead(1, 2) == 0
