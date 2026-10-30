"""Turn a captured phrase into 16 kHz audio Whisper can actually read.

Silence and room noise are not sent to the model. Whisper invents words
("you", "Bye.", "T-T-T...") when a quiet clip is amplified into fake speech.
"""

import re
from difflib import SequenceMatcher
from math import gcd
from typing import Optional

import numpy as np

TARGET_RATE = 16000
TARGET_PEAK = 0.95
MAX_GAIN = 6.0
# A 20ms frame quieter than this is room hiss, not a voice.
MIN_SPEECH_RMS = 0.005

_HALLUCINATED = {
    "you",
    "bye",
    "thanks for watching",
    "thank you for watching",
    "thanks for watching please subscribe",
    "subscribe",
    "subtitles by",
    "translated by",
}


def to_mono(indata) -> np.ndarray:
    """Keep the channel that actually contains the voice."""
    data = np.asarray(indata, dtype=np.float32)
    if data.ndim == 2 and data.shape[1] > 1:
        energy = np.mean(np.square(data), axis=0)
        data = data[:, int(np.argmax(energy))]
    return np.ascontiguousarray(np.reshape(data, -1), dtype=np.float32)


def prepare_speech_audio(audio, sample_rate: int) -> np.ndarray:
    """Anti-aliased resample to 16 kHz, then level the phrase.

    Returns an empty array when the clip is mostly silence, so the model
    is never asked to invent a transcript for it.
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    rate = int(sample_rate or TARGET_RATE)
    if not _has_speech(samples, rate):
        return np.zeros(0, dtype=np.float32)
    samples = _trim_silence(samples, rate)
    samples = _resample_16k(samples, rate)
    return _level(samples)


def transcribe_speech(model, audio_16k: np.ndarray, language: Optional[str], model_tag: str = "") -> str:
    """Decode one phrase. small.en is accurate with a single beam.

    Distilled checkpoints mishear words unless beam search is on, and
    that search stays under a second for them. A full medium model does
    not: it spends several seconds on the same phrase.
    """
    if audio_16k.size < int(TARGET_RATE * 0.35):
        return ""
    lang = None if not language or language == "auto" else language
    beam_size = 5 if "distil" in str(model_tag or "") else 1
    segments, _info = model.transcribe(
        np.ascontiguousarray(audio_16k, dtype=np.float32),
        language=lang,
        beam_size=beam_size,
        best_of=1,
        patience=1.0,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
        compression_ratio_threshold=2.2,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )
    parts = []
    for segment in segments:
        if float(getattr(segment, "no_speech_prob", 0) or 0) > 0.85:
            continue
        if float(getattr(segment, "avg_logprob", 0) or 0) < -1.1:
            continue
        if float(getattr(segment, "compression_ratio", 0) or 0) > 2.4:
            continue
        parts.append(segment.text)
    text = " ".join(parts).strip()
    if _looks_invented(text):
        return ""
    return text


def is_cross_talk_echo(previous_speaker: str, previous_text: str, speaker: str, text: str) -> bool:
    """True when the other source just captured the same sentence.

    The microphone and the meeting loopback often hear one voice twice.
    The second copy is dropped. A person repeating themselves is kept.
    """
    if not previous_text or not text or previous_speaker == speaker:
        return False
    previous_words = _words(previous_text)
    new_words = _words(text)
    if not previous_words or not new_words:
        return False
    previous_line = " ".join(previous_words)
    new_line = " ".join(new_words)
    if SequenceMatcher(None, previous_line, new_line).ratio() >= 0.72:
        return True
    shared = set(previous_words) & set(new_words)
    if not shared:
        return False
    return len(new_words) <= 2 and set(new_words) <= set(previous_words)


def take_speech_phrase(state: dict, chunk) -> Optional[np.ndarray]:
    """Return a finished utterance, or None while the person is still talking.

    Quiet has to last 0.7s before the phrase closes, so a breath does not
    cut the sentence. Sound also has to rise well above the room before it
    counts as speech, which keeps silence from being transcribed.
    """
    sr = int(state["sr"])
    samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return None

    rms = float(np.sqrt(np.mean(np.square(samples))))
    speaking = bool(state.get("is_speaking"))
    if not speaking:
        floor = float(state.get("noise_floor", 0.0015))
        state["noise_floor"] = (0.95 * floor) + (0.05 * rms)
    floor = max(float(state.get("noise_floor", 0.0015)), 1e-5)
    configured = float(state.get("cfg_thresh") or 0.008)
    # The saved silence setting can be higher than a quiet microphone.
    # Cap it so a real voice still opens a phrase, and ignore a single blip.
    ceiling = min(max(configured, 0.004), 0.01)
    open_at = max(0.004, min(ceiling, max(floor * 3.0, 0.004)))
    is_speech = rms >= (open_at * 0.5 if speaking else open_at)

    pre_roll = int(sr * 0.35)
    buffer = state.get("buffer")
    if buffer is None or len(buffer) == 0:
        buffer = np.zeros(0, dtype=np.float32)

    if not speaking:
        buffer = np.concatenate([buffer, samples])
        if len(buffer) > pre_roll:
            buffer = buffer[-pre_roll:]
        state["buffer"] = buffer
        if is_speech:
            state["speech_frames"] = int(state.get("speech_frames", 0)) + int(samples.size)
            if state["speech_frames"] >= int(sr * 0.2):
                state["is_speaking"] = True
                state["silent_frames"] = 0
        else:
            state["speech_frames"] = 0
        return None

    buffer = np.concatenate([buffer, samples])
    if is_speech:
        state["silent_frames"] = 0
    else:
        state["silent_frames"] = int(state.get("silent_frames", 0)) + int(samples.size)

    ended = state["silent_frames"] >= int(sr * 0.7) or len(buffer) >= int(sr * 8.0)
    if not ended:
        state["buffer"] = buffer
        return None

    phrase = np.ascontiguousarray(buffer, dtype=np.float32) if len(buffer) >= int(sr * 0.4) else None
    state["buffer"] = buffer[-pre_roll:] if len(buffer) > pre_roll else np.zeros(0, dtype=np.float32)
    state["is_speaking"] = False
    state["silent_frames"] = 0
    state["speech_frames"] = 0
    return phrase


def _has_speech(audio: np.ndarray, sample_rate: int) -> bool:
    """True when enough of the clip is louder than room hiss.

    A finished phrase is often speech from start to end, so it has no quiet
    stretch to compare against. Loudness is an absolute check.
    """
    frame = max(1, int(sample_rate * 0.02))
    usable = audio[: len(audio) - (len(audio) % frame)]
    if usable.size < frame * 8:
        return False
    rms = np.sqrt(np.mean(np.square(usable.reshape(-1, frame)), axis=1))
    if float(np.percentile(rms, 90)) < MIN_SPEECH_RMS:
        return False
    return float(np.mean(rms >= MIN_SPEECH_RMS)) >= 0.15


def _trim_silence(audio: np.ndarray, sample_rate: int, pad_sec: float = 0.12) -> np.ndarray:
    """Keep the voiced middle. A quiet tail is where Whisper adds a fake ending."""
    frame = max(1, int(sample_rate * 0.02))
    count = len(audio) // frame
    if count < 4:
        return audio
    usable = audio[: count * frame]
    rms = np.sqrt(np.mean(np.square(usable.reshape(count, frame)), axis=1))
    quiet = float(np.percentile(rms, 30))
    voiced = np.flatnonzero(rms >= max(quiet * 2.5, 0.006))
    if voiced.size == 0:
        return audio
    pad = int(pad_sec * sample_rate)
    start = max(0, int(voiced[0]) * frame - pad)
    end = min(len(audio), (int(voiced[-1]) + 1) * frame + pad)
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


def _looks_invented(text: str) -> bool:
    cleaned = " ".join(text.lower().replace(",", " ").split())
    cleaned = cleaned.strip(" .!?,;:\"'")
    if not cleaned or not any(char.isalnum() for char in cleaned):
        return True
    if cleaned in _HALLUCINATED:
        return True
    if re.search(r"(.\s*-\s*){6,}", text):
        return True
    words = _words(text)
    if len(words) >= 4 and len(set(words)) / len(words) < 0.34:
        return True
    return False


def _words(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())


def _resample_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_RATE:
        return np.ascontiguousarray(audio, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
        divisor = gcd(sample_rate, TARGET_RATE)
        # A sharp Kaiser window keeps stopband aliasing out without smearing
        # the bursts Whisper uses to tell consonants apart.
        converted = resample_poly(
            audio,
            TARGET_RATE // divisor,
            sample_rate // divisor,
            window=("kaiser", 14.0),
        )
        return np.ascontiguousarray(converted, dtype=np.float32)
    except Exception:
        target_len = max(1, int(len(audio) * TARGET_RATE / sample_rate))
        converted = np.interp(
            np.linspace(0, len(audio), target_len, endpoint=False),
            np.arange(len(audio)),
            audio,
        )
        return np.ascontiguousarray(converted, dtype=np.float32)


def _level(audio: np.ndarray) -> np.ndarray:
    """Bring speech up to a clear peak. A single click must not set the gain."""
    magnitude = np.abs(audio)
    if magnitude.size == 0:
        return audio
    peak = float(np.max(magnitude))
    robust_peak = float(np.quantile(magnitude, 0.999))
    if peak < 1e-4:
        return audio
    # Scale from the real peak so the waveform is not clipped. Fall back to
    # the robust peak only when a few samples dwarf the rest of the phrase.
    reference = robust_peak if peak > robust_peak * 4.0 else peak
    gain = min(TARGET_PEAK / max(reference, 1e-4), MAX_GAIN)
    return np.clip(audio * gain, -0.95, 0.95).astype(np.float32)
