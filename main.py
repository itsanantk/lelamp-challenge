"""Main demo loop: camera -> engagement -> behavior FSM -> lamp actuation,
composited into one window / one recorded video.

--chat runs the conversational recall agent (chat.py) on a background
thread against this SAME lamp instance and window, instead of it being a
separate process with its own lamp/window that only shared state through
the memory db file -- see docs/ARCHITECTURE.md for why that used to be
the design and what changed.

Usage:
    python main.py                       interactive demo window
    python main.py --chat --voice        also talk to it -- mic in, TTS out, same window
    python main.py --record out.mp4      also write the composite feed to a video file
    python main.py --label               enable ground-truth labeling for eval (press SPACE)
    python main.py --mute                disable sound cues
    python main.py --test-frames 60      headless smoke test, no window, no camera loop forever
    python main.py --auto-quit-after 8   interactive mode, but auto-quits after N seconds (scripted smoke tests)

Keys (interactive mode):
    q       quit
    SPACE   (with --label) toggle "I am currently looking at the lamp" ground truth
"""
from __future__ import annotations

import argparse
import csv
import datetime
import threading
import time

import cv2

import chat
import config
import viz
from behavior.adaptation import AdaptationEngine
from behavior.object_watch import ObjectWatcher, play_wave_back
from behavior.state_machine import BehaviorFSM, State
from lamp import SimulatedLamp
from memory.store import MemoryStore
from perception.audio_monitor import AudioActivityMonitor
from perception.engagement import EngagementPipeline
from perception.hand_wave import HandWaveDetector
from perception.vision_memory import VisionMemory


def _open_log_writer(enabled: bool):
    if not enabled:
        return None, None
    path = config.LOGS_DIR / f"session_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "t_ms", "face_found", "engaged_pred", "fsm_state", "label",
        "yaw", "pitch", "ema_yaw", "ema_pitch", "engagement_latency_ms",
        "yolo_latency_ms", "loop_latency_ms",
    ])
    print(f"[log] writing labeled session log to {path}")
    return f, writer


def run(args: argparse.Namespace) -> None:
    viz.enable_dpi_awareness()  # must happen before any window gets created -- see viz.py
    pipeline = EngagementPipeline()
    lamp = SimulatedLamp(mute=args.mute)
    fsm = BehaviorFSM(lamp=lamp)

    adapt_engine = None
    if not args.no_adapt:
        if args.fresh_adaptation and config.LOGS_DIR.joinpath("adaptation_state.json").exists():
            config.LOGS_DIR.joinpath("adaptation_state.json").unlink()
        adapt_engine = AdaptationEngine.load()
        adapt_engine.apply_to(fsm)

    store = None
    vision_memory = None
    object_watcher = None
    if not args.no_memory:
        store = MemoryStore(fresh=args.fresh_memory)
        vision_memory = VisionMemory(store)
        object_watcher = ObjectWatcher(lamp)

    audio_monitor = None
    if not args.no_audio_gate:
        audio_monitor = AudioActivityMonitor()
        audio_monitor.start()

    hand_wave = None
    if not args.no_hand_wave:
        if config.HAND_LANDMARKER_MODEL.exists():
            hand_wave = HandWaveDetector()
        else:
            print(f"[main] hand-wave detection disabled: {config.HAND_LANDMARKER_MODEL} not found "
                  f"(see README for the download command)")

    chat_thread = None
    chat_shutdown = None
    if args.chat:
        chat_shutdown = threading.Event()
        chat_args = argparse.Namespace(no_gui=True, voice=args.voice, mute=args.mute, ask=None,
                                        multi_user=args.multi_user, wake_word=args.wake_word)
        # store=None here on purpose -- MemoryAgent's own sqlite3 connection
        # has to be created on the thread that'll actually use it (see
        # chat.run's docstring), not handed in from this one.
        chat_thread = threading.Thread(
            target=chat.run, args=(chat_args,),
            kwargs={"lamp": lamp, "fsm": fsm, "vision_memory": vision_memory, "shutdown_event": chat_shutdown},
            daemon=True)
        chat_thread.start()

    log_file, log_writer = _open_log_writer(args.label or args.test_frames > 0)
    ground_truth_label = 0

    video_writer = None
    if args.record:
        config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.RECORDINGS_DIR / args.record if not args.record.count("/") and not args.record.count("\\") else args.record

    last_t = time.perf_counter()
    frame_count = 0
    interactive_elapsed = 0.0
    prev_fsm_state = fsm.state
    pan_bearing_ema = 0.0  # smoothed follower for viz.pan_crop_frame -- see config.CAMERA_PAN_FOLLOW_HZ
    info_panel_cache = None
    INFO_PANEL_REFRESH_EVERY = 4  # debug text panel is cheap to read stale for a few frames, expensive to redraw every frame at this resolution
    window_name = "LeLamp Challenge - webcam | lamp sim | fsm"
    window_ready = False

    try:
        while True:
            loop_t0 = time.perf_counter()
            frame, reading = pipeline.read()
            if frame is None:
                print("[main] camera read failed, stopping")
                break

            now = time.perf_counter()
            dt = now - last_t
            last_t = now

            engaged = reading.engaged if reading else False
            bearing = reading.user_bearing_deg if reading else None
            user_busy = audio_monitor.is_active() if audio_monitor is not None else False
            fsm_state = fsm.update(engaged=engaged, user_bearing_deg=bearing, dt=dt, user_busy=user_busy)
            lamp.update(dt)

            if adapt_engine is not None:
                if prev_fsm_state != State.ATTENTION_SEEKING and fsm_state == State.ATTENTION_SEEKING:
                    adapt_engine.on_attention_seek_started()
                if prev_fsm_state != State.ENGAGED and fsm_state == State.ENGAGED:
                    adapt_engine.on_engaged()
                adapt_engine.on_check_timeout()
                adapt_engine.apply_to(fsm)  # keep the FSM synced as the engine adapts mid-session
            prev_fsm_state = fsm_state

            if not args.no_camera_pan:
                # base_yaw is the same degrees-of-bearing convention as
                # pose_for_look_at's yaw_deg input (positive = user's
                # right), whatever subsystem last set the target pose --
                # tracking, attention-seeking, an engaged look, even idle
                # sway -- so reading it back is a simpler and more honest
                # "what is the lamp currently looking at" than trying to
                # separately track which subsystem currently owns the gaze.
                # Computed before the YOLO scan below (not just before
                # display) so both the detector's simulated FOV and the
                # on-screen pan follow the exact same bearing this tick.
                target_bearing = float(lamp.get_current_pose()[0])
                follow = min(1.0, dt * config.CAMERA_PAN_FOLLOW_HZ)
                pan_bearing_ema += (target_bearing - pan_bearing_ema) * follow

            yolo_latency_ms = None
            if vision_memory is not None:
                # Only object detection is restricted to the pan window --
                # engagement/face detection (above) always sees the full
                # frame. Those are conceptually two different sensing
                # channels here: whether a person is present and looking
                # at it is something a lamp would want to notice
                # regardless of which way its head happens to be turned;
                # "what objects are around" is what the FOV constraint is
                # actually meant to simulate (a camera mounted on the head
                # that can't see what it isn't pointed at).
                pan_zoom = config.CAMERA_PAN_ZOOM if not args.no_camera_pan else None
                scanned = vision_memory.maybe_scan(frame, pan_bearing_deg=pan_bearing_ema, pan_zoom=pan_zoom)
                if scanned is not None:  # only a real sample on ticks that actually ran YOLO
                    yolo_latency_ms = vision_memory.last_scan_latency_ms
                object_watcher.update(vision_memory.last_detections, fsm_state, dt, user_bearing_deg=bearing)
                vision_memory.fast_mode = object_watcher.should_scan_fast()

            if hand_wave is not None and fsm_state == State.ENGAGED:
                # Only while ENGAGED -- waving hello is a thing you do
                # when it's already paying attention to you, and it bounds
                # this third model's cost to exactly the situations where
                # a wave means anything (see config.py's own note).
                wave_bearing = hand_wave.update(frame, now)
                if wave_bearing is not None:
                    play_wave_back(lamp, wave_bearing)

            # Detection/engagement math above all runs on the raw,
            # un-flipped frame (that's what the corrected user-relative
            # bearing math in perception/ assumes) -- mirroring only
            # happens here, for what's actually drawn on screen.
            display_frame = viz.mirror_frame(frame)
            mirrored_detections = (viz.mirror_detections(vision_memory.last_detections, frame.shape[1])
                                    if vision_memory is not None else [])

            if not args.no_camera_pan:
                display_frame, crop_x0, crop_scale = viz.pan_crop_frame(
                    display_frame, pan_bearing_ema, config.CAMERA_HFOV_DEG, zoom=config.CAMERA_PAN_ZOOM)
                mirrored_detections = viz.remap_boxes_for_pan(mirrored_detections, crop_x0, crop_scale)

            webcam_hud = viz.draw_webcam_hud(display_frame, reading, fsm_state.name)
            if vision_memory is not None:
                webcam_hud = viz.draw_detections(webcam_hud, mirrored_detections)
            lamp_panel = lamp.render()

            # Rebuilding this panel (putText for ~20+ lines at 1920-wide
            # scale) is one of the costlier steps in the frame at this
            # resolution, and nobody reads debug text at 30fps -- refresh
            # it every few frames instead of on every single one.
            if info_panel_cache is None or frame_count % INFO_PANEL_REFRESH_EVERY == 0:
                info = fsm.debug_info()
                side_lines = [f"{k}: {v}" for k, v in info.items()]
                if adapt_engine is not None:
                    s = adapt_engine.summary()
                    rate = "n/a" if s["recent_response_rate"] is None else f"{s['recent_response_rate']:.0%}"
                    side_lines.append("")
                    side_lines.append(f"learned delay: {s['delay_s']:.1f}s  max tries: {s['max_attempts']}")
                    side_lines.append(f"response rate: {rate} ({s['samples']} samples)")
                if vision_memory is not None:
                    side_lines.append("")
                    side_lines.append("seen recently:")
                    for cls in store.list_known_classes()[:8]:
                        side_lines.append(f"  - {cls}")
                    if object_watcher.active:
                        side_lines.append("")
                        side_lines.append("watching tracked object...")
                if audio_monitor is not None and user_busy:
                    side_lines.append("")
                    side_lines.append("room busy (talking/media) --")
                    side_lines.append("holding off on attention-seeking")
                if args.label:
                    side_lines.append("")
                    side_lines.append(f"ground_truth: {'LOOKING' if ground_truth_label else 'away'} (SPACE to toggle)")
                # Full width of row 1 (webcam + lamp sim side by side), so the
                # info/debug row below spans the same width instead of being a
                # separate narrow column squeezed in next to them.
                row1_width = webcam_hud.shape[1] + lamp_panel.shape[1]
                info_panel_cache = viz.make_text_panel(row1_width, 500, side_lines, title="FSM + MEMORY + DEBUG",
                                                        columns=3, scale=1.8)
            composite = viz.compose_panels(webcam_hud, lamp_panel, info_panel_cache)

            loop_latency_ms = (time.perf_counter() - loop_t0) * 1000

            if log_writer is not None and reading is not None:
                log_writer.writerow([
                    reading.timestamp_ms, reading.face_found, int(reading.engaged), fsm_state.name,
                    ground_truth_label, reading.yaw_deg, reading.pitch_deg,
                    reading.ema_yaw_deg, reading.ema_pitch_deg, round(reading.latency_ms, 2),
                    round(yolo_latency_ms, 2) if yolo_latency_ms is not None else "",
                    round(loop_latency_ms, 2),
                ])

            if args.record:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(out_path), fourcc, 20.0,
                                                    (composite.shape[1], composite.shape[0]))
                video_writer.write(composite)

            frame_count += 1

            if args.test_frames > 0:
                if frame_count >= args.test_frames:
                    print(f"[test] processed {frame_count} frames headlessly, no crash. "
                          f"final fsm_state={fsm_state.name}")
                    break
                continue  # headless: skip imshow/waitKey entirely

            if not window_ready:
                # sized against the real composite dims on the first
                # interactive frame, not guessed from config constants --
                # see viz.open_fitted_window for why this needs to exist
                # at all (WINDOW_AUTOSIZE clipping off the right edge).
                viz.open_fitted_window(window_name, composite.shape[1], composite.shape[0])
                window_ready = True
            cv2.imshow(window_name, composite)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if args.label and key == ord(" "):
                ground_truth_label = 1 - ground_truth_label

            interactive_elapsed += dt
            if args.auto_quit_after is not None and interactive_elapsed >= args.auto_quit_after:
                print(f"[main] --auto-quit-after {args.auto_quit_after}s reached, stopping")
                break

    finally:
        if chat_thread is not None:
            # Signal + join before lamp.close() tears down the sound
            # worker -- letting the thread get killed mid sd.play()/TTS
            # call instead is exactly the bug class that used to segfault
            # this process (see lamp/sim_backend.py's _SoundWorker.stop()).
            chat_shutdown.set()
            chat_thread.join(timeout=20.0)
            if chat_thread.is_alive():
                print("[main] chat thread didn't stop in time (mid-LLM-call/TTS?), exiting anyway")
        pipeline.close()
        lamp.close()
        if audio_monitor is not None:
            audio_monitor.stop()
        if hand_wave is not None:
            hand_wave.close()
        if adapt_engine is not None:
            adapt_engine.save()
            print(f"[adapt] {adapt_engine.summary()}")
        if store is not None:
            store.close()
        if video_writer is not None:
            video_writer.release()
            print(f"[record] saved composite video to {out_path}")
        if log_file is not None:
            log_file.close()
        if args.test_frames == 0:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeLamp challenge demo")
    p.add_argument("--record", type=str, default=None, help="filename (in recordings/) to save composite video to")
    p.add_argument("--label", action="store_true", help="enable ground-truth labeling mode + CSV logging")
    p.add_argument("--mute", action="store_true", help="disable sound cues")
    p.add_argument("--test-frames", type=int, default=0, help="headless smoke-test mode: process N frames, no window")
    p.add_argument("--no-memory", action="store_true", help="disable YOLO scene memory formation")
    p.add_argument("--fresh-memory", action="store_true", help="wipe stored memory at startup")
    p.add_argument("--no-audio-gate", action="store_true",
                    help="disable interruption awareness (mic-based ambient audio gate)")
    p.add_argument("--no-adapt", action="store_true",
                    help="disable self-learning attention-seek timing (behavior/adaptation.py)")
    p.add_argument("--fresh-adaptation", action="store_true",
                    help="wipe learned attention-seek timing at startup")
    p.add_argument("--auto-quit-after", type=float, default=None,
                    help="interactive mode only: auto-quit after N seconds (for scripted smoke tests)")
    p.add_argument("--chat", action="store_true",
                    help="run the conversational recall agent on a background thread, sharing this lamp/window")
    p.add_argument("--voice", action="store_true", help="with --chat: talk instead of typing -- mic input + TTS output")
    p.add_argument("--multi-user", action="store_true",
                    help="with --chat --voice: identify which face was talking when more than one is in frame")
    p.add_argument("--wake-word", type=str, default=config.WAKE_WORD,
                    help=f'with --chat --voice: phrase that wakes it up to listen (default: "{config.WAKE_WORD}")')
    p.add_argument("--no-camera-pan", action="store_true",
                    help="disable the webcam view panning to follow the lamp's own gaze")
    p.add_argument("--no-hand-wave", action="store_true",
                    help="disable hand-wave detection (a third CPU model, only run while ENGAGED)")
    args = p.parse_args()
    if args.chat and not args.voice:
        # Typed chat blocks on input(), which no shutdown_event can reach
        # -- quitting the camera window would hang for the full join
        # timeout waiting on a thread stuck reading stdin. Voice mode's
        # own blocking calls all check shutdown_event already (see
        # conversation/voice.py). Text-only recall still works fine on
        # its own: `python chat.py`.
        p.error("--chat requires --voice here -- for typed-only recall, run chat.py standalone instead")
    return args


if __name__ == "__main__":
    run(parse_args())
