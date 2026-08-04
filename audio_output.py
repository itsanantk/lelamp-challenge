"""Shared coordination for the one real speaker both lamp/sim_backend.py's
sound cues and conversation/voice.py's TTS play through.

sounddevice's sd.play()/sd.wait() operate on a single, module-global
default output stream, not one scoped to each caller. Calling sd.play()
again before an earlier call's sd.wait() has returned doesn't queue
behind it -- it stops that playback immediately and starts the new one.
lamp/sim_backend.py's own _SoundWorker queues concurrent play_sound()
calls against each other just fine (one dedicated thread, one queue), and
conversation/voice.py's _speak_lock does the same for concurrent speak()
calls -- but neither knows about the other, since they're unrelated
subsystems that only happen to share this one piece of hardware. A
reminder's spoken message getting cut off mid-sentence by an
attention-seeking chirp landing on a different thread at the same moment
is exactly that gap, confirmed live. Every sd.play()+sd.wait() call in
this codebase should go through play_and_wait() below instead of calling
sounddevice directly, so the two subsystems queue behind each other
instead of racing for the same output slot.
"""
from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

_lock = threading.Lock()


def play_and_wait(samples: np.ndarray, samplerate: int) -> None:
    with _lock:
        sd.play(samples, samplerate=samplerate)
        sd.wait()
