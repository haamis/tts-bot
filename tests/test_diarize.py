"""Multi-voice !rvc diarization: clustering, merging, pitch assignment,
timeline reassembly, and profile caching. All model-dependent steps are
injected, so nothing here downloads weights or touches the network."""
import numpy as np
import pytest

from ttsbot.media import diarize
from ttsbot.media.audio import (
    concat_audio,
    load_mono,
    resample,
    slice_audio,
    write_wav,
)
from ttsbot.media.diarize import Segment


# --- audio helpers -----------------------------------------------------------


def sine(freq, seconds, sr=16000, amp=0.3):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_resample_changes_length_and_rate(tmp_path):
    audio = sine(440, 1.0, sr=44100)
    out = resample(audio, 44100, 16000)
    assert abs(out.size / 16000 - 1.0) < 0.01


def test_resample_same_rate_is_noop():
    audio = sine(440, 0.5)
    assert resample(audio, 16000, 16000) is audio


def test_slice_audio_clips_bounds():
    audio = sine(440, 2.0)
    assert slice_audio(audio, 16000, -1.0, 0.5).size == 8000
    assert slice_audio(audio, 16000, 1.5, 5.0).size == 8000
    assert slice_audio(audio, 16000, 1.0, 1.5).size == 8000


def test_concat_audio_with_gaps():
    sr = 16000
    parts = [sine(440, 0.1, sr=sr), sine(880, 0.1, sr=sr)]
    joined = concat_audio(parts, gap_s=0.1, sr=sr)
    assert joined.size == int(sr * (0.1 + 0.1 + 0.1))


def test_concat_audio_empty():
    assert concat_audio([]).size == 0
    assert concat_audio([np.zeros(0, dtype=np.float32)]).size == 0


def test_wav_roundtrip(tmp_path):
    path = str(tmp_path / "x.wav")
    write_wav(path, sine(440, 0.5), 16000)
    audio, sr = load_mono(path)
    assert sr == 16000
    assert audio.dtype == np.float32
    assert abs(audio.size / sr - 0.5) < 0.01


# --- VAD ---------------------------------------------------------------------


def test_voice_mask_detects_bursts():
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    audio = np.concatenate([silence, sine(440, 1.0, sr=sr), silence])
    mask = diarize.voice_mask(audio, sr)
    assert mask.size == audio.size // int(sr * 0.03)
    voiced_ratio = mask.mean()
    assert 0.2 < voiced_ratio < 0.6


def test_voiced_windows_split_on_speech():
    sr = 16000
    # two 3s speech islands separated by 3s of silence
    audio = np.concatenate([sine(220, 3.0, sr=sr), np.zeros(3 * sr, np.float32), sine(440, 3.0, sr=sr)])
    mask = diarize.voice_mask(audio, sr)
    windows = diarize.voiced_windows(mask, audio.size / sr)
    assert len(windows) >= 2
    assert windows[0][1] <= 3.5  # first island ends near 3s
    assert windows[-1][0] >= 2.5  # last island starts after the silence


# --- clustering ---------------------------------------------------------------


def far_apart_embeddings():
    """Two tight speaker groups, cosine-far apart."""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 0.1, (6, 16)) + np.array([1.0] + [0.0] * 15)
    b = rng.normal(0, 0.1, (6, 16)) + np.array([0.0] * 15 + [1.0])
    return np.vstack([a, b]).astype(np.float32)


def test_cluster_force_k():
    embs = far_apart_embeddings()
    labels, detected = diarize.cluster_embeddings(embs, k=2, threshold=0.0)
    assert detected == 2
    assert len(set(labels[:6])) == 1 and len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_cluster_threshold_collapses_to_two():
    embs = far_apart_embeddings()
    labels, detected = diarize.cluster_embeddings(embs, k=5, threshold=0.5)
    # the two groups are cosine-far; collapse must not merge them, and must
    # not invent 5 speakers out of 2
    assert detected == 2
    assert labels[0] != labels[6]


def test_cluster_threshold_0_always_k():
    embs = np.vstack([far_apart_embeddings(), far_apart_embeddings()[:4]])  # 16 windows
    labels, detected = diarize.cluster_embeddings(embs, k=4, threshold=0.0)
    assert detected == 4 and len(set(labels)) == 4


def test_cluster_single_window():
    labels, detected = diarize.cluster_embeddings(np.ones((1, 8), dtype=np.float32), 2, 0.5)
    assert labels.tolist() == [0] and detected == 1


def test_cluster_never_exceeds_k_when_threshold_too_high():
    embs = far_apart_embeddings()
    labels, detected = diarize.cluster_embeddings(embs, k=1, threshold=99.0)
    assert detected == 1 and len(set(labels)) == 1


def test_combined_distance_splits_by_pitch_when_embeddings_identical():
    """Male/female voices whose wav2vec2 embeddings interleave must still
    split when their pitches differ by an octave."""
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 16)
    embs = (base + rng.normal(0, 0.01, (8, 16))).astype(np.float32)
    f0s = [85.0, 250.0, 82.0, 240.0, 90.0, 260.0, 88.0, 230.0]
    labels, detected = diarize.cluster_embeddings(embs, k=2, threshold=0.35, f0s=f0s)
    assert detected == 2
    assert labels[0] == labels[2] == labels[4]  # male windows together
    assert labels[1] == labels[3] == labels[7]  # female windows together


def test_combined_distance_keeps_same_pitch_speakers_together():
    embs = far_apart_embeddings()
    f0s = [85.0] * 6 + [250.0] * 6
    labels, detected = diarize.cluster_embeddings(embs, k=2, threshold=0.35, f0s=f0s)
    assert detected == 2
    assert labels[0] != labels[6]


def test_combined_distance_untrusted_f0_falls_back_to_embedding():
    embs = far_apart_embeddings()
    f0s = [None] * 6 + [250.0] * 6
    labels, detected = diarize.cluster_embeddings(embs, k=2, threshold=0.35, f0s=f0s)
    assert detected == 2
    assert labels[0] != labels[6]


def test_combined_distances_no_pitch_is_pure_cosine():
    embs = far_apart_embeddings()
    d = diarize.combined_distances(embs, None)
    d2 = diarize.combined_distances(embs, [None] * 12)
    assert np.allclose(d, d2)


# --- merging ------------------------------------------------------------------


def test_merge_windows_merges_same_cluster():
    windows = [(0.0, 1.5), (2.0, 3.5), (4.0, 5.5)]
    labels = np.array([0, 0, 1])
    segments = diarize.merge_windows(windows, labels)
    assert [(s.start, s.end, s.cluster) for s in segments] == [
        (0.0, 3.5, 0),
        (4.0, 5.5, 1),
    ]


def test_merge_windows_drops_short_segments():
    windows = [(0.0, 1.5), (2.0, 3.5), (4.0, 4.3)]  # last one < 0.4s? 0.3s
    labels = np.array([0, 1, 0])
    segments = diarize.merge_windows(windows, labels)
    assert [(s.start, s.end) for s in segments] == [(0.0, 1.5), (2.0, 3.5)]


# --- audible gap absorption ---------------------------------------------------

SR = 16000


def _media_with_gap(gap_audio):
    """0-1s speech, gap, 3-4s speech."""
    return np.concatenate([sine(220, 1.0, sr=SR), gap_audio, sine(440, 1.0, sr=SR)])


def test_absorb_audible_gap_joins_preceding_segment():
    audio = _media_with_gap(sine(330, 2.0, sr=SR, amp=0.2))  # loud gap
    segments = [Segment(0.0, 1.0, 0), Segment(3.0, 4.0, 1)]
    out = diarize.absorb_audible_gaps(audio, segments)
    assert [(s.start, s.end, s.cluster) for s in out] == [(0.0, 3.0, 0), (3.0, 4.0, 1)]
    # input untouched
    assert (segments[0].start, segments[0].end) == (0.0, 1.0)


def test_absorb_keeps_silent_gap():
    audio = _media_with_gap(np.zeros(2 * SR, np.float32))
    segments = [Segment(0.0, 1.0, 0), Segment(3.0, 4.0, 1)]
    out = diarize.absorb_audible_gaps(audio, segments)
    assert [(s.start, s.end) for s in out] == [(0.0, 1.0), (3.0, 4.0)]


def test_absorb_ignores_leading_and_trailing_audio():
    audio = np.concatenate([
        sine(330, 1.0, sr=SR, amp=0.2),  # loud intro -> must stay original
        sine(220, 1.0, sr=SR),
        np.zeros(SR, np.float32),
        sine(440, 1.0, sr=SR),
        sine(330, 1.0, sr=SR, amp=0.2),  # loud outro -> must stay original
    ])
    segments = [Segment(1.0, 2.0, 0), Segment(3.0, 4.0, 1)]
    out = diarize.absorb_audible_gaps(audio, segments)
    assert [(s.start, s.end) for s in out] == [(1.0, 2.0), (3.0, 4.0)]


def test_absorb_quiet_gaps_below_threshold_stay_original():
    audio = _media_with_gap(sine(330, 2.0, sr=SR, amp=0.0005))  # ~-60dBFS
    segments = [Segment(0.0, 1.0, 0), Segment(3.0, 4.0, 1)]
    out = diarize.absorb_audible_gaps(audio, segments)
    assert [(s.start, s.end) for s in out] == [(0.0, 1.0), (3.0, 4.0)]


def test_absorb_no_segments_or_single():
    audio = sine(220, 1.0, sr=SR)
    assert diarize.absorb_audible_gaps(audio, []) == []
    out = diarize.absorb_audible_gaps(audio, [Segment(0.0, 0.5, 0)])
    assert [(s.start, s.end, s.cluster) for s in out] == [(0.0, 0.5, 0)]


# --- temporal propagation -----------------------------------------------------


def test_smooth_flips_expressive_fragment():
    """A blip whose pitch matches the FLANKING speaker is a fragment: flip."""
    labels = [0, 0, 1, 0, 0]
    win_f0 = [82, 85, 96, 88, 84]  # blip f0 ~ male range (both clusters male-ish)
    # own cluster (1) median 96; flank (0) median ~84.75 -> f0 closer to flank
    assert diarize.smooth_labels(labels, win_f0) == [0, 0, 0, 0, 0]


def test_smooth_keeps_genuine_short_interjection():
    """A,A,B,A,A with B's pitch matching B's cluster (which has other
    windows elsewhere): a real interjection must survive even though the
    temporal shape invites a flip."""
    labels = [0, 0, 1, 0, 0, 1, 1, 0]
    win_f0 = [82, 85, 250, 88, 84, 260, 240, 86]
    assert diarize.smooth_labels(labels, win_f0) == [0, 0, 1, 0, 0, 1, 1, 0]


def test_smooth_untrusted_pitch_falls_back_to_temporal_prior():
    labels = [0, 0, 1, 0, 0]
    win_f0 = [82, 85, None, 88, 84]
    assert diarize.smooth_labels(labels, win_f0) == [0, 0, 0, 0, 0]


def test_smooth_solo_cluster_blip_flips_on_temporal_prior():
    """A 1-window cluster has no sibling pitch evidence — the temporal
    prior decides (a lone window between two agreeing runs is treated as
    a glitch, not a speaker)."""
    labels = [0, 0, 1, 0, 0]
    win_f0 = [82, 85, 250, 88, 84]  # blip high-pitch but cluster 1 is solo
    assert diarize.smooth_labels(labels, win_f0) == [0, 0, 0, 0, 0]


def test_smooth_without_f0_keeps_old_behavior():
    assert diarize.smooth_labels([0, 0, 1, 0, 0]) == [0, 0, 0, 0, 0]
    # flanked by the same label on both sides -> flips
    assert diarize.smooth_labels([0, 1, 2, 1, 0]) == [0, 1, 1, 1, 0]


def test_smooth_ignores_none_and_boundaries():
    labels = [0, None, 1, None, 0]
    assert diarize.smooth_labels(labels, [80, None, 300, None, 80]) == [0, None, 1, None, 0]
    assert diarize.smooth_labels([1]) == [1]


def test_propagate_fills_from_nearest_trusted():
    # 2-slot gap, both sides same cluster -> filled
    assert diarize.propagate_labels([(0, 1)] * 4, [0, None, None, 0]) == [0, 0, 0, 0]
    # disagreement, equidistant -> unassigned
    assert diarize.propagate_labels([(0, 1)] * 4, [0, None, 1, 1]) == [0, None, 1, 1]
    # disagreement, nearer side wins
    assert diarize.propagate_labels([(0, 1)] * 5, [0, None, None, 1, 1]) == [0, 0, 1, 1, 1]


def test_propagate_conflict_tie_stays_unassigned():
    """A window equidistant between two speakers must not guess."""
    out = diarize.propagate_labels([(0, 1)] * 3, [0, None, 1])
    assert out == [0, None, 1]


def test_propagate_single_sided():
    out = diarize.propagate_labels([(0, 1)] * 3, [1, None, None])
    assert out == [1, 1, 1]
    out = diarize.propagate_labels([(0, 1)] * 3, [None, None, 0])
    assert out == [0, 0, 0]


def test_propagate_beyond_radius_stays_none():
    out = diarize.propagate_labels(
        [(0, 1)] * 8, [None, None, None, None, 0, 1, None, None]
    )
    assert out == [None, 0, 0, 0, 0, 1, 1, 1]
    # a >3-slot gap from BOTH neighbours stays original audio; the window
    # at the exact midpoint of a 7-slot gap is equidistant (4 vs 4) -> None
    out = diarize.propagate_labels(
        [(0, 1)] * 9, [0, None, None, None, None, None, None, None, 1]
    )
    assert out == [0, 0, 0, 0, None, 1, 1, 1, 1]


def test_merge_windows_skips_unassigned():
    windows = [(0.0, 1.5), (2.0, 3.5), (4.0, 5.5)]
    labels = [0, None, 1]
    segments = diarize.merge_windows(windows, labels)
    assert [(s.start, s.end, s.cluster) for s in segments] == [(0.0, 1.5, 0), (4.0, 5.5, 1)]


# --- pitch --------------------------------------------------------------------


def test_measure_f0_pure_tones():
    f0_low = diarize.measure_f0(sine(110, 2.0), 16000)
    f0_high = diarize.measure_f0(sine(250, 2.0), 16000)
    assert f0_low is not None and f0_high is not None
    assert 100 < f0_low < 125
    assert 230 < f0_high < 270
    assert f0_high > f0_low


def test_measure_f0_silence_returns_none():
    assert diarize.measure_f0(np.zeros(16000, np.float32), 16000) is None


def test_measure_f0_short_audio_returns_none():
    assert diarize.measure_f0(sine(110, 0.2), 16000) is None


def test_assign_voices_ranks_by_pitch():
    # cluster 1 measured higher pitch -> gets the higher-pitch voice
    mapping = diarize.assign_voices(
        f0s={0: 120.0, 1: 220.0},
        voices=["low", "high"],
        voice_f0={"low": 115.0, "high": 210.0},
        first_appearance=[0, 1],
    )
    assert mapping == {0: "low", 1: "high"}


def test_assign_voices_rank_preserved_outside_voice_range():
    """The reported bug: speakers 143/302Hz, voices snake 111.8/otacon 148.7.
    Rank must pair 143->snake, 302->otacon (closest-match crossed them)."""
    mapping = diarize.assign_voices(
        f0s={0: 143.0, 1: 302.0},
        voices=["snake", "otacon"],
        voice_f0={"snake": 111.8, "otacon": 148.7},
        first_appearance=[0, 1],
    )
    assert mapping == {0: "snake", 1: "otacon"}


def test_assign_voices_user_example_monotone():
    """Speakers 200/300Hz, voices A=100/B=200 -> A gets 200, B gets 300."""
    mapping = diarize.assign_voices(
        f0s={0: 200.0, 1: 300.0},
        voices=["a", "b"],
        voice_f0={"a": 100.0, "b": 200.0},
        first_appearance=[0, 1],
    )
    assert mapping == {0: "a", 1: "b"}


def test_assign_voices_swaps_when_pitches_oppose():
    mapping = diarize.assign_voices(
        f0s={0: 230.0, 1: 125.0},
        voices=["low", "high"],
        voice_f0={"low": 115.0, "high": 210.0},
        first_appearance=[0, 1],
    )
    assert mapping == {0: "high", 1: "low"}


def test_assign_voices_unmeasured_cluster_goes_last_by_appearance():
    mapping = diarize.assign_voices(
        f0s={0: None, 1: 220.0},
        voices=["low", "high"],
        voice_f0={"low": 115.0, "high": 210.0},
        first_appearance=[0, 1],
    )
    # cluster 1 (220Hz) matches "high" (210Hz); cluster 0 unknown -> "low"
    assert mapping == {0: "low", 1: "high"}


def test_assign_voices_unprofiled_voice_ranked_after_profiled():
    mapping = diarize.assign_voices(
        f0s={0: 120.0, 1: 220.0},
        voices=["mystery", "high"],
        voice_f0={"mystery": None, "high": 210.0},
        first_appearance=[0, 1],
    )
    # only "high" has a profile -> must land on the higher cluster
    assert mapping == {0: "mystery", 1: "high"}


def test_assign_voices_tie_keeps_appearance_order():
    mapping = diarize.assign_voices(
        f0s={0: 150.0, 1: 150.0},
        voices=["a", "b"],
        voice_f0={"a": 150.0, "b": 150.0},
        first_appearance=[1, 0],  # cluster 1 appeared first
    )
    assert mapping == {1: "a", 0: "b"}


# --- profiles -----------------------------------------------------------------


class FakeCfg:
    def __init__(self, model, pitch=0, f0_method="pm"):
        self.name = "x"
        self.rvc_model = model
        self.pitch = pitch
        self.f0_method = f0_method


def test_profile_cache_roundtrip_and_key_validation(tmp_path):
    path = str(tmp_path / "voice_pitch.json")
    model = tmp_path / "a.pth"
    model.write_bytes(b"fake")
    cfg = FakeCfg(str(model), pitch=-4, f0_method="pm")

    profiles = {}
    profiles["x"] = {"f0": 123.4, "key": diarize.profile_key(cfg, model.stat().st_mtime)}
    diarize.save_profiles(path, profiles)
    loaded = diarize.load_profiles(path)
    assert diarize.profile_f0(loaded, "x", cfg) == pytest.approx(123.4)

    # same model file, changed tuning -> stale
    assert diarize.profile_f0(loaded, "x", FakeCfg(str(model), pitch=2)) is None
    assert diarize.profile_f0(loaded, "x", FakeCfg(str(model), f0_method="rmvpe")) is None
    assert diarize.profile_f0(loaded, "x", FakeCfg(str(tmp_path / "other.pth"))) is None
    assert diarize.profile_f0(loaded, "missing", cfg) is None
    # missing model file -> stale
    assert diarize.profile_f0(loaded, "x", FakeCfg("/does/not/exist.pth")) is None


def test_profile_invalidated_by_model_mtime(tmp_path):
    import os

    model = tmp_path / "a.pth"
    model.write_bytes(b"fake")
    path = str(tmp_path / "voice_pitch.json")
    cfg = FakeCfg(str(model))
    profiles = {"x": {"f0": 100.0, "key": diarize.profile_key(cfg, model.stat().st_mtime)}}
    assert diarize.profile_f0(profiles, "x", cfg) == 100.0

    st = model.stat()
    os.utime(model, (st.st_atime, st.st_mtime + 10))
    assert diarize.profile_f0(profiles, "x", cfg) is None


def test_load_profiles_tolerates_missing_or_corrupt(tmp_path):
    assert diarize.load_profiles(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert diarize.load_profiles(str(bad)) == {}


# --- end-to-end pipeline with injected models ---------------------------------


def fake_embed_factory(cluster_of_window):
    """Deterministic embedder: window i gets unit vector at cluster index."""
    def embed(windows):
        out = np.zeros((len(windows), 8), dtype=np.float32)
        for i, _ in enumerate(windows):
            out[i, cluster_of_window[i] % 8] = 1.0
        return out

    return embed


def test_diarize_media_two_speakers(tmp_path):
    sr = 16000
    # speaker A (110Hz) 0-4s, silence, speaker B (250Hz) 7-11s
    audio = np.concatenate([
        sine(110, 4.0, sr=sr),
        np.zeros(3 * sr, np.float32),
        sine(250, 4.0, sr=sr),
        np.zeros(1 * sr, np.float32),
    ])
    src = tmp_path / "media.wav"
    write_wav(str(src), audio, sr)

    # window 0..1 land in island A, later windows in island B
    result = diarize.diarize_media(
        str(src),
        voices=["trump", "snake"],
        voice_f0={"trump": 115.0, "snake": 210.0},
        threshold=0.0,
        embed_fn=fake_embed_factory([0, 0, 1, 1, 1, 1]),
        f0_fn=diarize.measure_f0,  # real pyin on real tones
    )
    assert result.detected == 2
    assert len(result.segments) >= 2
    # pitch ranks: 250Hz cluster -> snake (210Hz), 110Hz cluster -> trump
    by_voice = {v: c for c, v in result.voice_of_cluster.items()}
    assert by_voice["trump"] != by_voice["snake"]
    f0_trump = result.cluster_f0[by_voice["trump"]]
    f0_snake = result.cluster_f0[by_voice["snake"]]
    assert f0_trump is not None and f0_snake is not None
    assert f0_snake > f0_trump


def test_analyze_finalize_matches_diarize_media(tmp_path):
    """The Phase 2 split is behavior-preserving: analyze + finalize locally
    must equal the monolithic path (server returns the analysis as JSON)."""
    sr = 16000
    audio = np.concatenate([
        sine(110, 4.0, sr=sr),
        np.zeros(3 * sr, np.float32),
        sine(250, 4.0, sr=sr),
        np.zeros(1 * sr, np.float32),
    ])
    src = tmp_path / "media.wav"
    write_wav(str(src), audio, sr)
    kwargs = dict(
        voices=["trump", "snake"],
        voice_f0={"trump": 115.0, "snake": 210.0},
        threshold=0.0,
    )
    full = diarize.diarize_media(
        str(src), **kwargs,
        embed_fn=fake_embed_factory([0, 0, 1, 1, 1, 1]),
        f0_fn=diarize.measure_f0,
    )
    analysis = diarize.analyze_media(
        str(src), 2, 0.0,
        embed_fn=fake_embed_factory([0, 0, 1, 1, 1, 1]),
        f0_fn=diarize.measure_f0,
    )
    split = diarize.finalize_result(analysis, str(src), kwargs["voices"], kwargs["voice_f0"])
    assert split.detected == full.detected
    assert [(s.start, s.end, s.cluster) for s in split.segments] == [
        (s.start, s.end, s.cluster) for s in full.segments
    ]
    assert split.voice_of_cluster == full.voice_of_cluster
    assert split.cluster_f0 == full.cluster_f0


def test_diarize_media_rebuild_timeline(tmp_path):
    sr = 16000
    audio = np.concatenate([sine(110, 3.0, sr=sr), np.zeros(2 * sr, np.float32), sine(250, 3.0, sr=sr)])
    src = tmp_path / "media.wav"
    write_wav(str(src), audio, sr)

    result = diarize.diarize_media(
        str(src),
        voices=["a", "b"],
        voice_f0={"a": None, "b": None},
        threshold=0.0,
        embed_fn=fake_embed_factory([0, 0, 1, 1]),
        f0_fn=lambda a, sr: 100.0,
    )
    # fake "converted" segments: constant tone per segment
    conv_paths = []
    for i, seg in enumerate(result.segments):
        p = tmp_path / f"conv_{i}.wav"
        write_wav(str(p), sine(330 + 100 * i, seg.end - seg.start, sr=sr), sr)
        conv_paths.append(str(p))

    out = tmp_path / "final.wav"
    diarize.rebuild_timeline(str(src), result, conv_paths, str(out))
    final, final_sr = load_mono(str(out))
    assert final_sr == sr
    # timeline roughly preserves total duration (converted len == slice len)
    assert abs(final.size / sr - audio.size / sr) < 0.3


def test_result_summary_formats_f0():
    result = diarize.DiarizationResult(
        segments=[Segment(0, 1, 0)],
        voice_of_cluster={0: "trump", 1: "snake"},
        cluster_f0={0: 122.4, 1: None},
        detected=2,
    )
    assert diarize.result_summary(result) == "trump @ 122Hz, snake @ ?Hz"


# --- tiny-cluster rejection (noise must not consume a voice) -------------------


def tiny_noise_media():
    """Male speech 0-4s, female 7-11s, loud noise burst at the very end."""
    sr = SR
    noise = (0.4 * np.sin(2 * np.pi * 900 * np.arange(int(0.8 * sr)) / sr)).astype(np.float32)
    return np.concatenate([
        sine(110, 4.0, sr=sr),
        np.zeros(3 * sr, np.float32),
        sine(250, 4.0, sr=sr),
        np.zeros(3 * sr, np.float32),
        noise,
    ])


def test_diarize_rejects_trailing_noise_cluster(tmp_path):
    src = tmp_path / "duo.wav"
    write_wav(str(src), tiny_noise_media(), SR)

    # Window order: 0,1 = male island; 2,3 = female island; 4 = the noise
    # burst, embedded far from both speakers (mirrors the real wav2vec2
    # behaviour on the reported clip).
    def embed(windows):
        out = np.zeros((len(windows), 8), np.float32)
        for i, _ in enumerate(windows):
            if i in (0, 1):
                out[i, 0] = 1.0
            elif i in (2, 3):
                out[i, 1] = 1.0
            else:
                out[i, 5] = 1.0
        return out

    result = diarize.diarize_media(
        str(src), ["snake", "otacon"], voice_f0={"snake": 115.7, "otacon": 166.5},
        threshold=0.35, embed_fn=embed, f0_fn=diarize.measure_f0,  # real pyin on real tones
    )
    # noise (< 1.5s of windows) must NOT be a speaker: exactly 2 detected
    assert result.detected == 2
    assert len(result.voice_of_cluster) == 2
    # male (110Hz) -> snake (115.7), female (250Hz) -> otacon (166.5)
    by_voice = {v: c for c, v in result.voice_of_cluster.items()}
    f0_snake = result.cluster_f0[by_voice["snake"]]
    f0_otacon = result.cluster_f0[by_voice["otacon"]]
    assert f0_snake is not None and f0_otacon is not None
    assert f0_snake < f0_otacon
