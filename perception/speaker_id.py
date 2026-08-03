"""Per-speaker voice identification: a lightweight voice-embedding
fingerprint (resemblyzer) for "whose voice is this," persisted across
turns *and* sessions -- distinct from perception/multi_face.py's
SpeakerDetector, which only ever answers "which face in frame was
talking this turn," with no memory of who that was from one turn to the
next. Voice-embedding based rather than face-recognition based because
the whole audio pipeline here (Whisper, tone analysis) already operates
on raw mic audio -- this reuses that same signal instead of adding a
face-recognition dependency on top of MediaPipe's landmark-only face
mesh (geometry, not an identity embedding).

Enrollment is automatic, not an explicit "who are you" flow: the first
time a voice is heard, it becomes a new speaker; every later utterance is
matched against known speakers by embedding cosine similarity, and the
matched speaker's stored embedding is updated as a running average (not
overwritten) so it drifts gently with natural voice variation instead of
freezing on whatever the very first sample happened to sound like.

Persisted as one small JSON file, not a new SQLite table -- a handful of
384-float embeddings plus a name/count/timestamp per speaker doesn't need
a database, and keeping it separate from memory/store.py's observation
log means wiping scene memory (--fresh-memory) doesn't also forget who
everyone is.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

import config


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class SpeakerIdentifier:
    def __init__(self, db_path: Path | None = None, similarity_threshold: float | None = None):
        self._db_path = db_path if db_path is not None else config.SPEAKER_DB
        self._threshold = similarity_threshold if similarity_threshold is not None else config.SPEAKER_MATCH_THRESHOLD
        self._encoder = None  # lazy -- constructing this pulls in resemblyzer/librosa,
                                # a real import + model-load cost not worth paying if
                                # speaker ID never actually gets used in a given run
        self.speakers: dict[str, dict] = {}
        self._load()

    def _ensure_encoder(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
        return self._encoder

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        data = json.loads(self._db_path.read_text())
        self.speakers = {sid: {**s, "embedding": np.array(s["embedding"], dtype=np.float32)}
                          for sid, s in data.items()}

    def _save(self) -> None:
        data = {sid: {**s, "embedding": s["embedding"].tolist()} for sid, s in self.speakers.items()}
        self._db_path.write_text(json.dumps(data))

    def identify(self, audio: np.ndarray, sample_rate: int) -> tuple[str, bool]:
        """Returns (speaker_id, is_new). Too little audio to carry a
        reliable embedding (see SPEAKER_MIN_AUDIO_S) returns a fresh
        throwaway id each time instead of risking a false match/enrollment
        off a near-silent or clipped clip."""
        if audio.size < sample_rate * config.SPEAKER_MIN_AUDIO_S:
            return f"unknown-{uuid.uuid4().hex[:8]}", True
        from resemblyzer import preprocess_wav
        encoder = self._ensure_encoder()
        wav = preprocess_wav(audio, source_sr=sample_rate)
        if wav.size < sample_rate * config.SPEAKER_MIN_AUDIO_S:
            # preprocess_wav trims silence -- a clip that was long enough
            # raw but is mostly silence can still end up too short after
            # that trim.
            return f"unknown-{uuid.uuid4().hex[:8]}", True
        embedding = encoder.embed_utterance(wav)
        return self._match_and_update(embedding)

    def _match_and_update(self, embedding: np.ndarray) -> tuple[str, bool]:
        """Split out from identify() so the matching/enrollment logic is
        testable with synthetic embeddings, without needing the real
        resemblyzer model in every test."""
        best_id, best_sim = None, -1.0
        for sid, s in self.speakers.items():
            sim = _cosine(embedding, s["embedding"])
            if sim > best_sim:
                best_id, best_sim = sid, sim

        if best_id is not None and best_sim >= self._threshold:
            s = self.speakers[best_id]
            n = s["seen_count"]
            s["embedding"] = (s["embedding"] * n + embedding) / (n + 1)
            s["seen_count"] = n + 1
            s["last_seen"] = time.time()
            self._save()
            return best_id, False

        new_id = f"speaker-{uuid.uuid4().hex[:8]}"
        self.speakers[new_id] = {"embedding": embedding, "name": None, "seen_count": 1, "last_seen": time.time()}
        self._save()
        return new_id, True

    def set_name(self, speaker_id: str, name: str) -> None:
        if speaker_id in self.speakers:
            self.speakers[speaker_id]["name"] = name
            self._save()

    def display_name(self, speaker_id: str) -> str:
        s = self.speakers.get(speaker_id)
        if s and s.get("name"):
            return s["name"]
        if speaker_id.startswith("unknown-"):
            return "someone"
        return f"speaker {speaker_id.split('-')[1]}"
