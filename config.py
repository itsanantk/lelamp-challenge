"""Paths and tuning constants live here so every module agrees on the
project root and so the thresholds that actually shape behavior aren't
buried inside the modules that use them.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
RECORDINGS_DIR = ROOT_DIR / "recordings"

FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"
YOLO_MODEL = MODELS_DIR / "yolo11s.pt"
PIPER_MODEL = MODELS_DIR / "en_US-amy-medium.onnx"
PIPER_MODEL_CONFIG = MODELS_DIR / "en_US-amy-medium.onnx.json"
MEMORY_DB = LOGS_DIR / "memory.sqlite3"

for _d in (MODELS_DIR, LOGS_DIR, RECORDINGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
# 1920x1080 is this webcam's actual max (confirmed by probing supported
# modes -- it doesn't do a higher-res 4:3 like 1280x960, only steps up
# through 16:9 modes), used instead of the original 640x480 for a
# noticeably bigger, sharper live window -- rendering natively at this
# resolution rather than upscaling a smaller capture into a bigger
# window, which would just be a blurrier version of the same detail.
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# No camera calibration step, so this is a guess at the webcam's horizontal
# FOV, just precise enough to turn "where in the frame" into a bearing angle.
CAMERA_HFOV_DEG = 60.0

# Camera panning (viz.pan_crop_frame): a digital crop/zoom of the display
# frame that follows the lamp's own current gaze bearing (its base_yaw
# joint), so the webcam view visually "pans" the way the lamp itself pans
# to look around a scene -- never real camera PTZ, and never touches the
# raw frame detection/engagement actually run on (see main.py).
CAMERA_PAN_ZOOM = 1.35        # how much the view zooms in while following -- bigger
                                # values pan further but crop out more of the edges
CAMERA_PAN_FOLLOW_HZ = 1.2    # smoothing rate on the followed bearing -- without this
                                # the view would jump with every small arm correction
                                # instead of panning smoothly

# --- Engagement hysteresis -----------------------------------------------
# ENTER_* is the tight cone that starts engagement (has to be clearly
# looking); EXIT_* is the looser cone that ends it (has to clearly look
# away). The gap between them is a deadband against boundary flicker, and
# the dwell-frame counts eat single-frame landmark noise on top of that.
ENGAGE_ENTER_YAW_DEG = 15.0
ENGAGE_ENTER_PITCH_DEG = 15.0
ENGAGE_EXIT_YAW_DEG = 25.0
ENGAGE_EXIT_PITCH_DEG = 25.0
ENGAGE_ENTER_DWELL_FRAMES = 5
ENGAGE_EXIT_DWELL_FRAMES = 8
ENGAGE_EMA_ALPHA = 0.35  # higher = less smoothing, more responsive

# --- Behavior timing -------------------------------------------------------
DISENGAGE_GRACE_S = 2.5          # brief hold before giving up on someone
ATTENTION_SEEK_DELAY_S = 4.0     # how long disengaged before trying to re-engage
ATTENTION_SEEK_MAX_ATTEMPTS = 3  # stop after this many tries
ATTENTION_SEEK_COOLDOWN_S = 6.0  # spacing between attempts -- doubled back up from an earlier
                                  # 3.0s pass at making this read as more persistent/dog-like,
                                  # which ended up feeling too naggy in practice

# --- Memory formation -------------------------------------------------------
# yolo11s over yolo11n: ~47 vs ~39.5 mAP on COCO, for +8ms/scan on this
# CPU (28ms vs 20ms at 320px) -- a meaningfully better hit rate for not
# much latency. yolo11m tested even better but ran 60-100ms/scan here,
# enough to show up as a visible stutter in the live window when a scan
# lands, so not worth it for a CPU-only interval scan. Bump YOLO_MODEL
# in config.py to yolo11m.pt if you want to trade a little smoothness for
# more accuracy.
#
# Scan intervals assume typical (plugged-in, not power-throttled) CPU
# performance -- measured live at ~28-90ms/scan on AC power (see
# docs/ARCHITECTURE.md). On battery, Windows' power plan can throttle this
# machine hard enough to more than double scan latency, which shows up as
# general sluggishness across the whole app, not just detection -- worth
# ruling out before re-tuning these if things still feel slow.
YOLO_CONF_THRESHOLD = 0.45
YOLO_SCAN_INTERVAL_S = 0.35       # normal cadence, interval-triggered (CPU only)
YOLO_FAST_SCAN_INTERVAL_S = 0.18  # cadence while actively watching a tracked object
YOLO_IMG_SIZE = 320

# Cheap frame-diff pre-check (perception/scene_change.py) skips a scan
# entirely when the scene hasn't visibly changed since the last one that
# ran -- only applies to normal-cadence scanning, not while fast-scanning
# a tracked object (see vision_memory.py). MAX_SKIPS forces a scan every
# so often regardless, so a genuinely static scene doesn't go stale.
SCENE_CHANGE_THRESHOLD = 0.02
SCENE_CHANGE_MAX_SKIPS = 6

# Tracked classes (see below) get their own, more lenient confidence bar
# instead of lowering YOLO_CONF_THRESHOLD globally -- a global drop would
# also let weaker, more ambiguous detections across all 80 COCO classes
# into scene memory. A phone or bottle is large and visually distinctive
# enough that a lower bar is low-risk specifically for them, and this is
# what actually helps the "doesn't detect at an angle" case: an off-angle
# phone is exactly the kind of detection that scores just under 0.45 but
# is still almost certainly a real phone.
TRACKED_CLASS_CONF_THRESHOLD = 0.30

# --- Object watching ("follow my phone when I put it down") ---------------
# Object classes that trigger the watch-and-remember behavior. COCO class
# names -- "cell phone" is what YOLO actually calls a phone; "bottle" is
# the closest COCO class to a water bottle (no separate "water bottle"
# class in the 80).
TRACKED_CLASSES = ["cell phone", "bottle"]
WATCH_REAIM_DEG = 4.0     # only issue a new pose command once it's moved at least
                           # this much since the last one -- avoids the arm jittering
                           # on frame-to-frame bbox noise while it's just sitting there
WATCH_LOST_GRACE_S = 2.0  # tolerate this long without a detection before treating it
                           # as actually gone and settling back -- rides out a few
                           # missed scans (a bad angle, brief occlusion) instead of
                           # the gesture flickering on and off with every miss
WATCH_FAST_SCAN_SETTLE_S = 1.5  # drop out of fast-scan mode once it's been this long
                                  # since the object last actually moved (not just been
                                  # visible) -- a stationary phone shouldn't pin the loop
                                  # at the fast interval forever
WATCH_STATIONARY_TIMEOUT_S = 3.0  # give up watching a still-visible-but-not-moving object
                                    # and return to a normal resting look after this long --
                                    # continuing to stare at something that stopped moving
                                    # reads as fixation, not attentiveness
REMINDER_PLACEMENT_SETTLE_S = 1.5  # behavior/reminders.py's own placement-confirmation
                                     # threshold -- used to just reuse WATCH_STATIONARY_TIMEOUT_S
                                     # above for a "consistent feel," but that constant answers
                                     # a different question ("when should I stop staring at
                                     # something that's done moving") than this one does ("how
                                     # long before I trust this position as a stable baseline
                                     # to detect a disturbance against"). Kept short on purpose:
                                     # every object_check reminder is gated behind this before
                                     # the deadline check even gets a bearing to point at, and
                                     # before disturbance-watching can start at all -- live
                                     # testing showed the full 3.0s (compounded with however
                                     # long it took to even get an initial sighting) reading as
                                     # a confusingly long silence right after asking for a watch.
WATCH_REACQUIRE_MARGIN = 2.5  # after giving up on a stationary object, require this many
                                # times WATCH_REAIM_DEG of movement to re-acquire -- otherwise
                                # a tiny bump on an otherwise-still phone would restart the
                                # whole acquire flourish, which reads as twitchy
WATCH_WAVE_MIN_REVERSALS = 3   # this many back-and-forth direction changes within the window
                                 # below counts as a wave, not just ordinary repositioning
WATCH_WAVE_WINDOW_S = 2.0       # how recent those direction changes need to be

# --- Idle look-around (behavior/idle_scan.py) -------------------------------
# Deliberately much longer than ATTENTION_SEEK_DELAY_S -- that timeline is
# about re-engaging a person who just left; this is a "nothing's happened
# in a while, let's glance around for anything new" behavior that should
# feel occasional, not restless.
IDLE_SCAN_DELAY_S = 14.0
IDLE_SCAN_HOLD_S = 2.0  # long enough for a real scan or two to land at each waypoint
                          # (see YOLO_SCAN_INTERVAL_S) before moving to the next one

# --- Ambient light reactivity (perception/ambient_light.py) ----------------
# Opt-in (--ambient-light), not opt-out -- and tuned to actually be
# noticeable when someone turns it on to see it work, not a background
# effect subtle enough to miss entirely. The original pass here (2.0s
# interval, 0.15 smoothing, 0.18 max nudge) took ~15-20s to converge and
# topped out as a barely-visible shift -- correct in principle, unverifiable
# in practice. This is the version someone can cover the camera with a
# hand and actually watch happen.
AMBIENT_LIGHT_SAMPLE_INTERVAL_S = 1.0  # room lighting doesn't change fast, but this
                                          # still shouldn't take forever to catch up
AMBIENT_LIGHT_SMOOTHING = 0.35  # EMA alpha on the luma reading -- still smooths out a
                                   # single-frame blip (samples are already 1s apart),
                                   # but converges in a handful of seconds, not ~20
AMBIENT_LIGHT_MIDPOINT = 0.35  # "normal room" reference luma (0..1) -- tuned lower than a
                                  # naive 0.5, since an indoor room read through a webcam
                                  # typically averages darker than that
AMBIENT_LIGHT_RESPONSE = 0.5   # how strongly a luma deviation from the midpoint maps to
                                  # a brightness nudge
AMBIENT_LIGHT_MAX_NUDGE = 0.35   # cap -- a real, visible swing now that this is
                                    # something the user explicitly turned on to see

# --- Hand-wave detection (perception/hand_wave.py) -------------------------
# A real hand, not the tracked object's own bearing -- MediaPipe's
# HandLandmarker, run at a modest gated interval and only while ENGAGED
# (waving hello is a thing you do when it's already paying attention to
# you), not every frame -- a third full model pass on top of the
# always-on FaceLandmarker and interval-scanned YOLO is real added CPU
# cost on top of what's already a tight budget (see vision_memory.py's own
# docstring on thread oversubscription).
HAND_SCAN_INTERVAL_S = 0.12       # ~8Hz -- fast enough to catch a few back-and-forth
                                    # reversals inside WATCH_WAVE_WINDOW_S
HAND_WAVE_MIN_DELTA = 0.02         # normalized-x wrist movement (0..1 frame width) between
                                    # scans small enough to ignore as noise, not a real swing

# --- Conversational recall -------------------------------------------------
# "anthropic" or "openai" -- whichever key you actually have credit on.
LLM_PROVIDER = os.environ.get("LELAMP_LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_MODEL = "claude-sonnet-5"
OPENAI_MODEL = "gpt-4o-mini"

# --- Voice I/O ---------------------------------------------------------
# Both directions run locally so voice mode doesn't depend on which LLM
# key is funded. small.en over base.en trades a bigger one-time download
# (~460MB vs ~140MB) and somewhat higher per-turn latency for meaningfully
# better transcription accuracy -- base.en was mishearing plain speech
# often enough in testing to be the complaint, not just the wake word.
WHISPER_MODEL = "small.en"
# A separate, much smaller model just for wake-word chunks (see
# _check_wake_chunk) -- decoding a 1-2s clip on small.en was the dominant
# cost in "why does it take a while to notice I'm talking to it" (small.en
# chosen for WHISPER_MODEL specifically for its better accuracy on real
# questions, at a real latency cost that doesn't matter there but does
# here). tiny.en only has to catch one distinctive word ("lamp"), not
# transcribe a full sentence accurately, so the accuracy tradeoff that
# ruled tiny.en out for WHISPER_MODEL doesn't apply to this narrower job.
# ~75MB vs small.en's ~460MB -- a real but small extra download/RAM cost
# for a meaningfully snappier wake response.
WAKE_WHISPER_MODEL = "tiny.en"
VOICE_SAMPLE_RATE = 16000

# Piper's own speaking-rate knob (see conversation/voice.py's _speak_piper),
# separate from SAPI5's "rate" property below -- Piper is the primary TTS
# path when its model files are present, SAPI5 is only the fallback, and
# they don't share a control (this one scales phoneme duration, not wpm).
# The model's own default is 1.0 (see en_US-amy-medium.onnx.json's
# inference.length_scale); live testing found that read too slow/deliberate
# for back-and-forth conversation. Lower = faster. 0.85 not chosen by ear
# alone -- it's a bounded first cut, same caveat as every other threshold in
# this file: raise it back toward 1.0 if it starts reading as rushed rather
# than brisk.
PIPER_LENGTH_SCALE = 0.85

# Wake-word gated instead of a blind fixed-length recording window: the
# mic passively polls in short chunks, only transcribing a chunk (via the
# same local Whisper model) once it actually contains sound, and only
# starts recording the real question once WAKE_WORD shows up in one of
# those chunks. Keeps mic activity from being interpreted as a question
# at random, and means you don't have to time your speech into an
# arbitrary window that starts the instant the script gets there.
WAKE_WORD = "hey lamp"
WAKE_CHUNK_S = 1.8   # kept short-ish so the wake phrase rarely spans a chunk boundary
                      # (and when it does, wait_for_wake_word's prev_text concatenation
                      # still catches it) -- shorter means less buffering delay before a
                      # chunk is even checked at all, which is most of "takes a while to
                      # notice I'm talking to it" alongside WAKE_WHISPER_MODEL above
VOICE_SILENCE_TIMEOUT_S = 1.2   # stop recording the question after this much continuous silence
VOICE_NO_SPEECH_TIMEOUT_S = 3.0  # bail out this fast if nothing at all was said after waking
                                  # (e.g. a false-positive wake trigger) -- otherwise this case
                                  # would run all the way to VOICE_MAX_RECORD_S with no benefit
VOICE_MAX_RECORD_S = 12.0       # hard cap so a stuck-open mic can't record forever

# After a wake+answer, chat.py stays listening for a follow-up instead of
# requiring the wake word again -- this is both the no-speech patience and
# the max_s ceiling for that follow-up listen (see chat.py's _listen).
CONVERSATION_FOLLOWUP_TIMEOUT_S = 60.0

# RMS level (float32 samples, [-1, 1]) above which a voice-mode chunk
# counts as "something worth transcribing," separate from
# AUDIO_GATE_RMS_THRESHOLD below -- they look like the same kind of
# number but calibrate two different situations. AUDIO_GATE_RMS_THRESHOLD
# is tuned for detecting ambient room activity (talking, TV) from the
# lamp's position on a desk; this one is tuned for actually-trying-to-
# talk-to-it speech into a laptop's built-in mic. 0.02 turned out to be
# well above real speech at normal distance/volume on a laptop mic --
# live-captured RMS values while saying the wake phrase peaked around
# 0.010-0.016 and never once crossed 0.02, so every chunk was silently
# discarded no matter how clearly it was spoken. 0.008 sits comfortably
# below those observed peaks and above the ambient noise floor seen in
# the same session (mostly 0.0000-0.0066).
VOICE_GATE_RMS_THRESHOLD = 0.008

# The "yelling" TEXT LABEL floor -- conversation/emotion.py's _classify(),
# judged after the fact on peak-RMS over the full recorded clip. This used
# to also gate the live jerk-back/startled reaction (conversation/
# voice.py's on_loud callback) -- split apart (see VOICE_FLINCH_RMS_
# THRESHOLD below) because the two need different sensitivities: the
# physical flinch is a "did a loud tone happen at all" trip-wire that
# should fire often and immediately, while the "yelling" label is a
# rarer, more extreme classification. Sharing one number meant raising it
# to stop the flinch mis-firing on merely-raised speech also silently
# raised the bar for the text label past where it should sit, and vice
# versa. Well above VOICE_GATE_RMS_THRESHOLD and conversation/emotion.py's
# own _LOUD_RMS ("tense"/"energetic" territory). 0.032 (the original
# value) fired on ordinary talking -- live-measured RMS while just
# speaking normally sometimes reached exactly that. 0.04 (tried next) was
# still too close to that same normal-talking peak to leave real margin.
VOICE_YELL_RMS_THRESHOLD = 0.5

# The live jerk-back/startled reaction's own floor (conversation/voice.py's
# on_loud callback, checked synchronously off raw audio mid-recording --
# see that callback's own docstring for why this is deliberately a dumb
# immediate trip-wire, not a tone judgment, not VOICE_YELL_RMS_THRESHOLD
# above). Also reused by conversation/emotion.py's _ANGRY_RMS as the point
# where the "tense"/"energetic" text classification stops blending in
# speaking rate and trusts loudness alone -- a live physical reaction and
# the text label a moment later should agree on when a loud tone actually
# happened. Set above VOICE_GATE_RMS_THRESHOLD/normal-talking peaks (up to
# ~0.032, see VOICE_YELL_RMS_THRESHOLD's own comment) but far below
# VOICE_YELL_RMS_THRESHOLD itself -- a startle reaction should trip on a
# genuinely raised voice, not require the same extremity as the "yelling"
# text label.
VOICE_FLINCH_RMS_THRESHOLD = 0.07

# --- Interruption awareness -------------------------------------------------
# RMS level (float32 samples, [-1, 1]) above which the mic counts as
# "something's going on" -- talking, TV, music. This shares the same
# physical mic as VOICE_GATE_RMS_THRESHOLD above (one laptop mic, no
# separate lamp hardware), so it should never have been calibrated
# separately -- it was, and it inherited the same bug that constant was
# already fixed for: 0.02 sits above the real-speech peaks this mic
# actually produces (~0.010-0.016, see VOICE_GATE_RMS_THRESHOLD's
# comment), so ordinary talking never crossed it and interruption
# awareness was silently a no-op since it was added. Matched to
# VOICE_GATE_RMS_THRESHOLD's already-validated value instead of picking
# a new one -- same mic, same room, same "comfortably below real speech,
# above the ambient floor" reasoning applies.
AUDIO_GATE_RMS_THRESHOLD = 0.008

# How long is_active() keeps reporting "busy" after the last loud block --
# equivalently, how many consecutive silent seconds it takes to actually
# count as quiet again. This IS the "wait N seconds of continuous silence"
# gate (see AudioActivityMonitor's own docstring), it was just tuned too
# short: 0.6s doesn't survive a normal conversational breath/pause, so
# talking to someone else with even a one-beat gap read as "quiet" and let
# attention-seeking start mid-conversation. 3.0s tolerates a real pause
# without tolerating an actually-finished conversation for long.
AUDIO_GATE_SUSTAIN_S = 3.0
