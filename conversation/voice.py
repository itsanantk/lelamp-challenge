"""Voice I/O for chat.py: mic -> local Whisper -> text, and text -> TTS.

Both directions run locally (Whisper via openai-whisper, TTS via pyttsx3
on top of SAPI5) so voice mode doesn't depend on which LLM key is funded,
and doesn't add a network round trip on top of the one the LLM call
already needs. Whisper gets its audio as an in-memory float32 array
straight from sounddevice rather than a file path, which sidesteps
needing ffmpeg installed (openai-whisper only shells out to ffmpeg when
you hand it a file path).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd

import config

_whisper_model = None


def preload() -> None:
    """Load the Whisper model up front so the first turn of a conversation
    isn't the one eating the ~10-15s load time."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f"[voice] loading Whisper ({config.WHISPER_MODEL})...")
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)
        print("[voice] ready")


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


def _peak_rms(audio: np.ndarray, samplerate: int, window_s: float = 0.25) -> float:
    """Max RMS over short subframes rather than one RMS over the whole
    chunk -- a short burst of speech in an otherwise-quiet WAKE_CHUNK_S
    window reads as much louder here than as a flat average, which was
    the actual cause of real speech landing just under the gate."""
    window = max(1, int(window_s * samplerate))
    if audio.size <= window:
        return _rms(audio)
    return max(_rms(audio[start:start + window]) for start in range(0, audio.size, window))


def record(duration_s: float, samplerate: int = config.VOICE_SAMPLE_RATE) -> np.ndarray:
    """Fixed-length blocking recording -- the building block used for each
    short wake-word poll. Not used for the actual question anymore (see
    record_until_silence), since a fixed window either cuts off a long
    answer or leaves dead air after a short one."""
    audio = sd.rec(int(duration_s * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def record_until_silence(max_s: float = config.VOICE_MAX_RECORD_S,
                          silence_s: float = config.VOICE_SILENCE_TIMEOUT_S,
                          no_speech_timeout_s: float = config.VOICE_NO_SPEECH_TIMEOUT_S,
                          samplerate: int = config.VOICE_SAMPLE_RATE,
                          stop_event: threading.Event | None = None) -> np.ndarray:
    """Records from the mic until speech has clearly started and then
    stopped (silence_s of continuous quiet after some sound was heard), or
    max_s is hit as a safety cap. If nothing is said at all -- e.g. a
    false-positive wake-word trigger from background noise -- bails out
    after just no_speech_timeout_s instead of waiting for the full max_s,
    since there's nothing to gain from waiting longer on pure silence. If
    stop_event is given, it's set the moment recording actually stops, so
    a caller sampling something else (e.g. video, for speaker ID) over the
    same window knows to stop too instead of running to its own separate
    timeout."""
    threshold = config.VOICE_GATE_RMS_THRESHOLD
    chunks: list[np.ndarray] = []
    lock = threading.Lock()
    state = {"speech_started": False, "silence_since": None}
    start_t = time.monotonic()
    done = threading.Event()

    def _callback(indata, frames, time_info, status) -> None:
        with lock:
            chunks.append(indata.copy())
        rms = _rms(indata)
        now = time.monotonic()
        if rms > threshold:
            state["speech_started"] = True
            state["silence_since"] = None
        elif state["speech_started"]:
            if state["silence_since"] is None:
                state["silence_since"] = now
            elif now - state["silence_since"] > silence_s:
                done.set()
        elif now - start_t > no_speech_timeout_s:
            done.set()

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=_callback):
        t0 = time.monotonic()
        while not done.is_set() and (time.monotonic() - t0) < max_s:
            sd.sleep(30)

    if stop_event is not None:
        stop_event.set()

    with lock:
        if not chunks:
            return np.zeros(0, dtype="float32")
        return np.concatenate(chunks, axis=0).flatten()


def transcribe(audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    preload()
    result = _whisper_model.transcribe(audio, fp16=False, language="en")
    return result["text"].strip()


_QUIT_WORDS = ("quit", "exit")
_printed_input_device = False


def _print_input_device_once() -> None:
    """Prints which mic sounddevice will actually record from. Only ever
    needs saying once per process, not once per chunk. This exists
    because "nothing is happening" has a failure mode that RMS numbers
    alone don't diagnose: if the OS's default input device isn't the mic
    you're actually talking into -- a Bluetooth headset that's paired but
    not selected, a webcam mic instead of the one you're near, whatever --
    every chunk reads as near-silent not because the gate is miscalibrated
    but because the right microphone was never being listened to at all."""
    global _printed_input_device
    if _printed_input_device:
        return
    _printed_input_device = True
    try:
        info = sd.query_devices(kind="input")
        print(f'[voice] recording from: "{info["name"]}" -- if that is not the mic '
              f'you are talking into (e.g. a Bluetooth headset that is paired but not '
              f'selected), that is very likely why nothing is being heard')
    except Exception as e:
        print(f"[voice] couldn't query the input device: {e}")


def wait_for_wake_word(wake_word: str = config.WAKE_WORD, chunk_s: float = config.WAKE_CHUNK_S,
                        samplerate: int = config.VOICE_SAMPLE_RATE) -> str:
    """Blocks until wake_word or a quit word is heard, and returns which
    ("wake" or "quit"). Recognizing quit here too -- not just as an answer
    to "what's your question" -- matters because without it there'd be no
    way to leave voice mode without first successfully waking it up: a
    wake-gated loop with no quit path of its own is a dead end short of
    Ctrl-C. Polls in short fixed chunks rather than a real always-on
    wake-word engine (no extra heavy dependency, reuses the Whisper model
    already loaded for real questions) -- each chunk only gets transcribed
    if it actually had sound in it, so idle silence doesn't cost a Whisper
    call.

    Two things that make matching more forgiving than a literal substring
    check on the whole phrase: only the *last word* of wake_word actually
    has to show up (for "hey lamp" that's "lamp") -- "hey" is a common
    filler word Whisper can mangle or drop for no real gain in
    distinctiveness, while "lamp" is unusual enough on its own to rarely
    false-trigger. And each chunk's transcript is checked concatenated
    with the previous one, not alone, so a phrase split across a chunk
    boundary (the end of one recording, the start of the next) still
    matches instead of silently going unheard on both sides of the split.
    Every chunk prints its RMS level, silent or not -- not just the ones
    that pass the volume gate and get transcribed. That matters because a
    gate set wrong (too strict for your mic's actual gain/distance) fails
    *silently*: every chunk gets skipped before transcription ever runs,
    so there'd be nothing to print if only non-silent chunks logged
    anything -- indistinguishable from the loop not running at all. Kept
    to one short line per chunk on purpose (just the number, no restated
    threshold/explanation -- that's already in the one-time line printed
    below) since this repeats continuously while idle: verbose per-chunk
    logging is exactly the kind of output that's cheap to print and
    expensive to have pasted back for debugging. Numbers that never get
    close to the threshold shown below mean the gate itself is the
    problem (fix: lower config.VOICE_GATE_RMS_THRESHOLD); numbers that
    cross it but a transcript that's never the right word means it's a
    Whisper accuracy or matching problem instead."""
    preload()
    _print_input_device_once()
    print(f'[voice] say "{wake_word}" to talk to it, or "quit" to exit... '
          f'(volume gate: {config.VOICE_GATE_RMS_THRESHOLD})')
    match_word = wake_word.lower().split()[-1]
    prev_text = ""
    while True:
        audio = record(chunk_s, samplerate=samplerate)
        rms = _peak_rms(audio, samplerate)
        if rms < config.VOICE_GATE_RMS_THRESHOLD:
            print(f'[voice] rms={rms:.4f}')
            prev_text = ""  # a real silent gap -- nothing here could be one half of a split phrase
            continue
        text = transcribe(audio).lower()
        print(f'[voice] rms={rms:.4f} heard: "{text}"')
        combined = f"{prev_text} {text}".strip()
        if match_word in combined:
            return "wake"
        if any(q in combined for q in _QUIT_WORDS):
            return "quit"
        prev_text = text


def listen() -> str:
    return transcribe(record_until_silence())


def speak(text: str) -> None:
    if not text:
        return
    import pyttsx3
    engine = pyttsx3.init()
    # SAPI5's default ~200wpm reads as rushed/choppy, especially over
    # Bluetooth output where short audio segments are prone to the same
    # stream-continuity glitches observed with the lamp's own sound cues.
    engine.setProperty("rate", 175)
    engine.say(text)
    engine.runAndWait()
