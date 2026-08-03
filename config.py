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

# A real yell's RMS floor -- shared between conversation/voice.py (fires
# the jerk-back/startled reaction live, mid-recording, off the raw audio
# callback) and conversation/emotion.py (labels the whole utterance
# "yelling" after the fact). One shared number so "did that count as a
# yell" means the same thing whether it's judged in real time on a single
# callback block or after the fact on peak-RMS over the full clip. Well
# above VOICE_GATE_RMS_THRESHOLD and above conversation/emotion.py's own
# _LOUD_RMS ("tense"/"energetic" territory) -- tuned by ear, same caveat
# as every other threshold in this file.
VOICE_YELL_RMS_THRESHOLD = 0.032

# --- Interruption awareness -------------------------------------------------
# RMS level (float32 samples, [-1, 1]) above which the mic counts as
# "something's going on" -- talking, TV, music. Measured ambient noise
# floor on this machine was ~0.00003; this leaves well over 2 orders of
# magnitude of headroom. Mic gain/distance varies by setup, so treat this
# as a starting point, not a calibrated constant.
AUDIO_GATE_RMS_THRESHOLD = 0.02
AUDIO_GATE_SUSTAIN_S = 0.6  # how long "recently active" holds true after the last loud block
