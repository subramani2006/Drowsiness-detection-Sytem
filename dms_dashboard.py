# dms_ultra_advanced_no_server.py
"""
DMS ULTRA — Advanced Final (Standalone Version)
Features:
- Drowsiness (EAR) and Yawn (MAR) detection using MediaPipe FaceMesh + OpenCV
- In-app ThingSpeak telemetry (periodic and alert-only modes)
- Snapshot on alert + manual snapshot
- Video recording (AVI)
- Auto-calibration (3s)
- Profiles (save/load driver settings)
- Alarm playback (WAV), volume and mute
- TTS feedback on recovery
- Export logs to CSV and PDF (FPDF optional)
- Toggle Light/Dark theme (persisted)
- Performance modes (High/Balance/Low)
- Cyberpunk Neon UI using CustomTkinter
- Robust error handling, logs, and graceful shutdown

Removed: External Mobile Alert Server dependency.

Save as: dms_ultra_advanced.py
Requirements:
pip install opencv-python mediapipe numpy scipy simpleaudio pyttsx3 Pillow customtkinter requests fpdf
"""

import os
import sys
import json
import time
import math
import threading
import csv
import traceback
import platform
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import simpleaudio as sa
import pyttsx3
import customtkinter as ctk
from PIL import Image, ImageTk, ImageEnhance
from tkinter import filedialog, messagebox
import tkinter as tk

# optional PDF export
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# safe mediapipe import
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except Exception as e:
    mp = None
    mp_face = None
    print("mediapipe not available or failed to initialize:", e)

# ---------------- Paths & config ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
SNAP_DIR = SCRIPT_DIR / "snapshots"
LOG_FILE = SCRIPT_DIR / "detection_log.csv"
CONFIG_FILE = SCRIPT_DIR / "config.json"
PROFILES_DIR = SCRIPT_DIR / "profiles"
DEFAULT_ALARM = SCRIPT_DIR / "SK-Madharasi-Glimpse-BGM.wav"
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

DEFAULTS = {
    "ear_thresh": 0.24,
    "eye_closed_sec": 2.0,
    "mar_thresh": 0.60,
    "thingspeak_key": "",
    "thingspeak_interval": 1.0,
    "thingspeak_mode": "periodic",  # 'periodic' or 'alerts'
    "alarm_path": str(DEFAULT_ALARM),
    "appearance": "dark",
    "alarm_volume": 1.0,
    "mute": False,
    "record": False,
    "performance": "balanced",  # 'high', 'balanced', 'low'
    "save_profiles": True
}

def load_config():
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            cfg = DEFAULTS.copy()
            cfg.update(data)
            return cfg
    except Exception as e:
        print("Config load error:", e)
    return DEFAULTS.copy()

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print("Config save error:", e)

config = load_config()

# ---------------- Utilities ----------------
def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def append_log(event):
    ts = now_ts()
    line = f"{ts} | {event}"
    print(line)
    try:
        new = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "event"])
            w.writerow([ts, event])
    except Exception as e:
        print("Log write error:", e)
    return line

def save_snapshot(frame, tag="snapshot"):
    try:
        fn = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}.jpg"
        path = SNAP_DIR / fn
        cv2.imwrite(str(path), frame)
        append_log(f"Snapshot saved: {path}")
        return str(path)
    except Exception as e:
        append_log(f"Snapshot error: {e}")
        return None

# small euclidean if scipy not present
def _dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(eye):
    try:
        A = _dist(eye[1], eye[5])
        B = _dist(eye[2], eye[4])
        C = _dist(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C > 1e-6 else 0.0
    except Exception:
        return 0.0

def mouth_aspect_ratio(mouth):
    try:
        A = _dist(mouth[1], mouth[5])
        B = _dist(mouth[2], mouth[4])
        C = _dist(mouth[0], mouth[3])
        return (A + B) / (2.0 * C) if C > 1e-6 else 0.0
    except Exception:
        return 0.0

# ---------------- ThingSpeak ----------------
def send_thingspeak(ear, mar, status):
    key = config.get("thingspeak_key","").strip()
    if not key:
        return False
    if not hasattr(send_thingspeak, "last"):
        send_thingspeak.last = 0.0
    try:
        interval = max(0.1, float(config.get("thingspeak_interval", 1.0)))
    except Exception:
        interval = 1.0
    now = time.time()
    if now - send_thingspeak.last < interval:
        return False
    send_thingspeak.last = now
    params = {"api_key": key, "field1": round(ear,3), "field2": round(mar,3), "field3": status}
    try:
        r = requests.get("https://api.thingspeak.com/update", params=params, timeout=4)
        if r.status_code == 200:
            append_log("ThingSpeak: update OK")
            return True
        else:
            append_log(f"ThingSpeak HTTP {r.status_code}")
    except Exception as e:
        append_log(f"ThingSpeak error: {e}")
    return False

# ---------------- Alarm Player ----------------
class AlarmPlayer:
    def __init__(self, wav_path=None):
        self.wav = wav_path or config.get("alarm_path") or str(DEFAULT_ALARM)
        self._stop = threading.Event()
        self.thread = None
        self._mute = config.get("mute", False)
        self._volume = float(config.get("alarm_volume", 1.0))

    def set_wav(self, path):
        self.wav = path
        config["alarm_path"] = path
        save_config(config)

    def set_volume(self, v):
        self._volume = float(v)
        config["alarm_volume"] = self._volume
        save_config(config)

    def set_mute(self, flag):
        self._mute = bool(flag)
        config["mute"] = self._mute
        save_config(config)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def _play_loop(self):
        if not os.path.exists(self.wav):
            append_log("Alarm file missing: " + str(self.wav))
            return
        try:
            wave_obj = sa.WaveObject.from_wave_file(self.wav)
        except Exception as e:
            append_log("Alarm load error: " + str(e))
            return
        while not self._stop.is_set():
            if self._mute:
                time.sleep(0.2)
                continue
            try:
                play = wave_obj.play()
                while play.is_playing():
                    if self._stop.is_set():
                        try: play.stop()
                        except: pass
                        break
                    time.sleep(0.02)
            except Exception as e:
                append_log("Alarm play error: " + str(e))
                break

    def stop(self):
        self._stop.set()

# ---------------- Detector Thread ----------------
class Detector(threading.Thread):
    def __init__(self, ui_callback, cam_index=0):
        super().__init__(daemon=True)
        self.ui_callback = ui_callback
        self.cam_index = cam_index
        self.cap = None
        self.running = False
        self.alarm = AlarmPlayer(config.get("alarm_path"))
        self.ear_buf = deque(maxlen=6)
        self.mar_buf = deque(maxlen=6)
        self.eye_close_since = None
        self.drowsy = False
        self.yawn = False
        self.fps = 0.0
        self._frame_cnt = 0
        self._fps_ts = time.time()
        self.last_ts_push = 0.0

        self.record_writer = None
        self.recording = config.get("record", False)

    def start_camera(self):
        # attempt various backends
        opened = False
        try:
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            opened = self.cap.isOpened()
        except Exception:
            try:
                self.cap = cv2.VideoCapture(self.cam_index)
                opened = self.cap.isOpened()
            except Exception:
                opened = False
        if not opened:
            raise RuntimeError(f"Cannot open camera index {self.cam_index}")
        # set resolution for better face detection; balance per performance mode
        perf = config.get("performance","balanced")
        if perf == "high":
            w,h = 1280,720
        elif perf == "low":
            w,h = 640,360
        else:
            w,h = 960,540
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        except:
            pass
        self.running = True
        self.start()

    def run(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue
                h, w = frame.shape[:2]
                ear = 0.0
                mar = 0.0
                if mp_face is not None:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = mp_face.process(rgb)
                        if results and getattr(results, "multi_face_landmarks", None):
                            lm = results.multi_face_landmarks[0].landmark
                            def P(i): return (int(lm[i].x * w), int(lm[i].y * h))
                            LEFT = [33,160,158,133,153,144]
                            RIGHT = [362,385,387,263,373,380]
                            MOUTH = [78,81,13,311,308,14]
                            left_eye = [P(i) for i in LEFT]
                            right_eye = [P(i) for i in RIGHT]
                            mouth = [P(i) for i in MOUTH]
                            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                            mar = mouth_aspect_ratio(mouth)
                            # draw markers small
                            for (x,y) in left_eye + right_eye + mouth:
                                cv2.circle(frame, (x,y), 1, (0,255,0), -1)
                    except Exception as e:
                        # Mediapipe error, ignore landmarks this frame
                        pass
                # smoothing buffers
                if ear > 0: self.ear_buf.append(ear)
                if mar > 0: self.mar_buf.append(mar)
                avg_ear = float(np.mean(self.ear_buf)) if len(self.ear_buf) else ear
                avg_mar = float(np.mean(self.mar_buf)) if len(self.mar_buf) else mar
                # FPS calc
                self._frame_cnt += 1
                now = time.time()
                if now - self._fps_ts >= 1.0:
                    self.fps = self._frame_cnt / (now - self._fps_ts)
                    self._frame_cnt = 0
                    self._fps_ts = now
                # detection logic
                ear_thresh = float(config.get("ear_thresh", DEFAULTS["ear_thresh"]))
                closed_sec = float(config.get("eye_closed_sec", DEFAULTS["eye_closed_sec"]))
                mar_thresh = float(config.get("mar_thresh", DEFAULTS["mar_thresh"]))
                status = "Normal"
                # drowsiness
                if avg_ear > 0 and avg_ear < ear_thresh:
                    if self.eye_close_since is None:
                        self.eye_close_since = time.time()
                    else:
                        if time.time() - self.eye_close_since >= closed_sec and not self.drowsy:
                            self.drowsy = True
                            status = "DROWSY"
                            append_log("Drowsiness detected")
                            save_snapshot(frame, "drowsy")
                            self.alarm.start()
                            # thingspeak if configured
                            try:
                                if config.get("thingspeak_mode","periodic") == "alerts":
                                    threading.Thread(target=send_thingspeak, args=(avg_ear, avg_mar, status), daemon=True).start()
                            except:
                                pass
                else:
                    # recover
                    self.eye_close_since = None
                    if self.drowsy:
                        self.drowsy = False
                        status = "Recovered"
                        self.alarm.stop()
                        append_log("Recovered from drowsiness")
                        try:
                            engine = pyttsx3.init()
                            engine.say("Please take rest and go")
                            engine.runAndWait()
                        except:
                            pass
                # yawn detection
                if avg_mar > mar_thresh:
                    if not self.yawn:
                        self.yawn = True
                        status = "YAWN"
                        append_log("Yawn detected")
                        save_snapshot(frame, "yawn")
                        self.alarm.start()
                        try:
                            if config.get("thingspeak_mode","periodic") == "alerts":
                                threading.Thread(target=send_thingspeak, args=(avg_ear, avg_mar, status), daemon=True).start()
                        except:
                            pass
                else:
                    if self.yawn:
                        self.yawn = False
                        self.alarm.stop()
                        try:
                            engine = pyttsx3.init()
                            engine.say("Please take rest and go")
                            engine.runAndWait()
                        except:
                            pass
                # thingSpeak periodic
                try:
                    if config.get("thingspeak_mode","periodic") == "periodic":
                        now_ts_local = time.time()
                        ts_int = max(0.1, float(config.get("thingspeak_interval", 1.0)))
                        if now_ts_local - self.last_ts_push >= ts_int:
                            threading.Thread(target=send_thingspeak, args=(avg_ear, avg_mar, status), daemon=True).start()
                            self.last_ts_push = now_ts_local
                except Exception:
                    pass
                # handle recording
                if config.get("record", False):
                    if self.record_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        outpath = SCRIPT_DIR / f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                        try:
                            self.record_writer = cv2.VideoWriter(str(outpath), fourcc, 20.0, (w,h))
                            append_log(f"Recording started: {outpath}")
                        except Exception as e:
                            append_log(f"Record start error: {e}")
                            self.record_writer = None
                    if self.record_writer is not None:
                        try:
                            self.record_writer.write(frame)
                        except:
                            pass
                else:
                    if self.record_writer:
                        try:
                            self.record_writer.release()
                        except:
                            pass
                        self.record_writer = None
                # callback to UI
                try:
                    self.ui_callback(frame, avg_ear, avg_mar, status, self.fps)
                except Exception as e:
                    # UI might be closed
                    pass
                time.sleep(0.01)
            except Exception as e:
                append_log("Detector loop error: " + str(e))
                time.sleep(0.1)
        # cleanup
        try:
            if self.cap: self.cap.release()
            if self.record_writer: self.record_writer.release()
        except:
            pass

    def ui_callback(self, frame, ear, mar, status, fps):
        # placeholder if direct callback needed
        pass

    def stop(self):
        self.running = False
        try:
            self.alarm.stop()
        except:
            pass

# ---------------- UI Application ----------------
ctk.set_appearance_mode(config.get("appearance","dark"))
ctk.set_default_color_theme("dark-blue")

class UltraApp:
    def __init__(self, root):
        self.root = root
        root.title("DMS ULTRA — Advanced")
        root.geometry("1368x820")
        root.resizable(False, False)

        # background canvas (animated)
        self.bg = tk.Canvas(root, bg="#000000", highlightthickness=0)
        self.bg.place(x=0, y=0, width=1368, height=820)
        self.bg_phase = 0.0
        self._animate_bg()

        # main frame
        self.main = ctk.CTkFrame(root, width=1280, height=760, corner_radius=16)
        self.main.place(x=44, y=28)

        # Left: video container
        self.video_frame = ctk.CTkFrame(self.main, width=840, height=600, corner_radius=14)
        self.video_frame.place(x=20, y=20)

        # neon canvas behind video
        self.neon_canvas = tk.Canvas(self.video_frame, bg="#041016", highlightthickness=0)
        self.neon_canvas.place(x=0, y=0, width=840, height=600)
        self.neon_pulse = 0.0
        self._draw_neon()

        # video label must be on top
        self.video_label = tk.Label(self.video_frame, bg="#000000")
        self.video_label.place(x=10, y=10, width=820, height=580)
        self.video_label.lift()

        # small overlay indicators at top-left of video
        self.indicator_frame = ctk.CTkFrame(self.video_frame, width=220, height=40, corner_radius=8)
        self.indicator_frame.place(x=12, y=12)
        self.lbl_rec = ctk.CTkLabel(self.indicator_frame, text="REC: OFF", text_color="#ff4444")
        self.lbl_rec.place(x=8, y=6)
        self.lbl_alarm = ctk.CTkLabel(self.indicator_frame, text="ALARM: OFF", text_color="#ff4444")
        self.lbl_alarm.place(x=100, y=6)

        # Right panel controls
        self.panel = ctk.CTkFrame(self.main, width=390, height=600, corner_radius=12)
        self.panel.place(x=880, y=20)

        ctk.CTkLabel(self.panel, text="DMS ULTRA", font=("Roboto",18,"bold")).place(x=20, y=12)
        self.btn_theme = ctk.CTkButton(self.panel, text="Toggle Light/Dark", width=180, command=self.toggle_theme)
        self.btn_theme.place(x=200, y=12)

        # Gauges
        self.gauge = tk.Canvas(self.panel, width=350, height=200, bg="#02060a", highlightthickness=0)
        self.gauge.place(x=20, y=50)

        # LED bar
        self.led = tk.Canvas(self.panel, width=350, height=36, bg="#02060a", highlightthickness=0)
        self.led.place(x=20, y=260)

        # Stats
        self.lbl_ear = ctk.CTkLabel(self.panel, text="EAR: -")
        self.lbl_ear.place(x=20, y=305)
        self.lbl_mar = ctk.CTkLabel(self.panel, text="MAR: -")
        self.lbl_mar.place(x=20, y=335)
        self.lbl_fps = ctk.CTkLabel(self.panel, text="FPS: -")
        self.lbl_fps.place(x=20, y=365)

        # Buttons
        self.btn_start = ctk.CTkButton(self.panel, text="Start", width=140, command=self.start_detection)
        self.btn_stop  = ctk.CTkButton(self.panel, text="Stop", width=140, fg_color="#e74c3c", command=self.stop_detection)
        self.btn_start.place(x=20, y=400)
        self.btn_stop.place(x=190, y=400)

        self.btn_cal = ctk.CTkButton(self.panel, text="Auto-Calibrate (3s)", width=320, command=self.auto_calibrate)
        self.btn_cal.place(x=20, y=445)

        # Sliders for threshold
        ctk.CTkLabel(self.panel, text="EAR Threshold").place(x=20, y=490)
        self.slider_ear = ctk.CTkSlider(self.panel, from_=0.12, to=0.4, width=320)
        self.slider_ear.set(config.get("ear_thresh", DEFAULTS["ear_thresh"]))
        self.slider_ear.place(x=20, y=515)
        ctk.CTkLabel(self.panel, text="MAR Threshold").place(x=20, y=545)
        self.slider_mar = ctk.CTkSlider(self.panel, from_=0.4, to=1.2, width=320)
        self.slider_mar.set(config.get("mar_thresh", DEFAULTS["mar_thresh"]))
        self.slider_mar.place(x=20, y=570)

        # Alarm selection and controls
        self.lbl_alarm_file = ctk.CTkLabel(self.panel, text=f"Alarm: {os.path.basename(config.get('alarm_path'))}")
        self.lbl_alarm_file.place(x=20, y=610)
        ctk.CTkButton(self.panel, text="Choose Alarm", width=160, command=self.choose_alarm).place(x=20, y=640)
        ctk.CTkButton(self.panel, text="Play Alarm", width=120, command=self.play_alarm_once).place(x=200, y=640)

        # Volume & mute
        ctk.CTkLabel(self.panel, text="Alarm Volume").place(x=20, y=680)
        self.vol_slider = ctk.CTkSlider(self.panel, from_=0, to=100, width=320, command=self.on_volume_change)
        self.vol_slider.set(int(float(config.get("alarm_volume",1.0))*100))
        self.vol_slider.place(x=20, y=705)
        self.mute_var = tk.BooleanVar(value=bool(config.get("mute", False)))
        self.mute_chk = ctk.CTkCheckBox(self.panel, text="Mute Alarm", variable=self.mute_var, command=self.on_mute_toggle)
        self.mute_chk.place(x=20, y=740)

        # ThingSpeak controls (in-app)
        ctk.CTkLabel(self.panel, text="ThingSpeak Key").place(x=20, y=780)
        self.ts_entry = ctk.CTkEntry(self.panel, width=220)
        self.ts_entry.insert(0, config.get("thingspeak_key",""))
        self.ts_entry.place(x=20, y=805)
        ctk.CTkButton(self.panel, text="Save TS Key", width=100, command=self.save_thingspeak_key).place(x=250, y=805)
        ctk.CTkLabel(self.panel, text="Mode (periodic/alerts)").place(x=20, y=835)
        self.ts_mode = ctk.CTkComboBox(self.panel, values=["periodic","alerts"], width=160)
        self.ts_mode.set(config.get("thingspeak_mode","periodic"))
        self.ts_mode.place(x=200, y=835)
        ctk.CTkLabel(self.panel, text="Interval (s)").place(x=20, y=865)
        self.ts_interval = ctk.CTkEntry(self.panel, width=80)
        self.ts_interval.insert(0, str(config.get("thingspeak_interval",1.0)))
        self.ts_interval.place(x=120, y=865)
        self.ts_status_label = ctk.CTkLabel(self.panel, text=("TS: ON" if config.get("thingspeak_key") else "TS: OFF"), text_color=("#76FF03" if config.get("thingspeak_key") else "#FF5252"))
        self.ts_status_label.place(x=220, y=865)

        # Profiles: save/load driver profiles
        ctk.CTkLabel(self.panel, text="Profiles").place(x=20, y=965)
        self.profile_list = ctk.CTkComboBox(self.panel, values=self._list_profiles(), width=220)
        self.profile_list.place(x=20, y=990)
        ctk.CTkButton(self.panel, text="Save Profile", width=120, command=self.save_profile).place(x=250, y=990)
        ctk.CTkButton(self.panel, text="Load Profile", width=120, command=self.load_profile).place(x=250, y=1025)

        # Bottom: graph and log
        self.graph_canvas = tk.Canvas(self.main, width=1240, height=160, bg="#021016", highlightthickness=0)
        self.graph_canvas.place(x=20, y=640)
        self.log_box = tk.Text(self.main, bg="#001015", fg="#bfefff")
        self.log_box.place(x=1020, y=640, width=280, height=160)
        self.log_box.configure(state="disabled")

        # Histories
        self.ear_hist = deque(maxlen=800)
        self.mar_hist = deque(maxlen=800)

        # Detector
        self.detector = None
        self.last_update_ts = 0.0

        # save last frame for snapshot
        self._last_frame = None

        # update UI tick
        self.root.after(200, self._ui_tick)

        append_log("DMS ULTRA Advanced (Standalone) initialized")

    # ---------------- UI Helpers ----------------
    def _animate_bg(self):
        try:
            w, h = 1368, 820
            t = getattr(self, "bg_phase", 0.0)
            a1 = np.array([6, 8, 25]) + (np.sin(t * 0.9) * 60)
            a2 = np.array([40, 2, 90]) + (np.cos(t * 0.7) * 80)
            cx = int(w * 0.45 + math.sin(t * 0.5) * 160)
            cy = int(h * 0.35 + math.cos(t * 0.4) * 120)
            yy, xx = np.mgrid[0:h, 0:w]
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            dmax = math.sqrt(w * w + h * h)
            mix = np.clip(d / dmax, 0, 1)[:, :, None]
            img = (a1 * (1 - mix) + a2 * mix).astype(np.uint8)
            # subtle scanlines
            for row in range(0, img.shape[0], 8):
                img[row:row+1, :, :] = np.clip(img[row:row+1, :, :] + (np.sin(t * 12) * 6), 0, 255)
            im = Image.fromarray(img)
            self.bg_imgtk = ImageTk.PhotoImage(im)
            try:
                self.bg.create_image(0, 0, image=self.bg_imgtk, anchor="nw")
            except:
                pass
            self.bg_phase = t + 0.03
        except Exception:
            pass
        self.root.after(80, self._animate_bg)

    def _draw_neon(self):
        try:
            c = self.neon_canvas
            c.delete("all")
            w, h = 840, 600
            pulse = (math.sin(self.neon_pulse) + 1) / 2.0
            r = int(80 + pulse * 160)
            g = int(120 + (1 - pulse) * 60)
            b = int(200 + pulse * 40)
            maincol = '#%02x%02x%02x' % (r, g, b)
            for i in range(6, 0, -1):
                rr = max(0, r - i * 6)
                gg = max(0, g - i * 4)
                bb = max(0, b - i * 3)
                col = '#%02x%02x%02x' % (rr, gg, bb)
                pad = i * 4
                c.create_rectangle(6 - pad, 6 - pad, w - 6 + pad, h - 6 + pad, outline=col, width=2)
            c.create_rectangle(6, 6, w - 6, h - 6, outline=maincol, width=4)
            self.neon_pulse += 0.08
        except Exception:
            pass
        try:
            self.root.after(90, self._draw_neon)
        except:
            pass

    def toggle_theme(self):
        cur = ctk.get_appearance_mode()
        new = "light" if cur == "dark" else "dark"
        ctk.set_appearance_mode(new)
        config["appearance"] = new
        save_config(config)
        append_log("Theme changed to " + new)

    def choose_alarm(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if p:
            config["alarm_path"] = p
            save_config(config)
            self.lbl_alarm_file.configure(text=f"Alarm: {os.path.basename(p)}")
            append_log("Alarm set: " + p)

    def play_alarm_once(self):
        def _play():
            try:
                path = config.get("alarm_path", str(DEFAULT_ALARM))
                if not os.path.exists(path):
                    append_log("Alarm file missing: " + path)
                    return
                obj = sa.WaveObject.from_wave_file(path)
                obj.play().wait_done()
            except Exception as e:
                append_log("Play alarm error: " + str(e))
        threading.Thread(target=_play, daemon=True).start()

    def on_volume_change(self, val):
        try:
            v = float(val) / 100.0
            config["alarm_volume"] = v
            save_config(config)
        except:
            pass

    def on_mute_toggle(self):
        config["mute"] = bool(self.mute_var.get())
        save_config(config)

    def save_thingspeak_key(self):
        key = self.ts_entry.get().strip()
        config["thingspeak_key"] = key
        config["thingspeak_mode"] = self.ts_mode.get()
        try:
            config["thingspeak_interval"] = float(self.ts_interval.get())
        except:
            config["thingspeak_interval"] = 1.0
        save_config(config)
        self.ts_status_label.configure(text=("TS: ON" if key else "TS: OFF"), text_color=("#76FF03" if key else "#FF5252"))
        append_log("ThingSpeak key saved")

    def _list_profiles(self):
        try:
            files = [f.stem for f in PROFILES_DIR.glob("*.json")]
            return files
        except:
            return []

    def save_profile(self):
        name = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data = {
            "ear_thresh": float(self.slider_ear.get()),
            "mar_thresh": float(self.slider_mar.get()),
            "eye_closed_sec": float(config.get("eye_closed_sec", DEFAULTS["eye_closed_sec"])),
            "alarm_path": config.get("alarm_path"),
            "appearance": config.get("appearance")
        }
        path = PROFILES_DIR / f"{name}.json"
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            append_log("Profile saved: " + str(path))
            self.profile_list.configure(values=self._list_profiles())
        except Exception as e:
            append_log("Profile save error: " + str(e))

    def load_profile(self):
        p = self.profile_list.get()
        if not p:
            messagebox.showinfo("Load profile", "Select a profile")
            return
        path = PROFILES_DIR / f"{p}.json"
        if not path.exists():
            messagebox.showerror("Load profile", "Profile not found")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.slider_ear.set(data.get("ear_thresh", config.get("ear_thresh")))
            self.slider_mar.set(data.get("mar_thresh", config.get("mar_thresh")))
            if data.get("alarm_path"):
                config["alarm_path"] = data.get("alarm_path")
                save_config(config)
                self.lbl_alarm_file.configure(text=f"Alarm: {os.path.basename(config['alarm_path'])}")
            append_log("Profile loaded: " + str(path))
        except Exception as e:
            append_log("Profile load error: " + str(e))

    # ---------------- Control functions ----------------
    def start_detection(self):
        if self.detector and getattr(self.detector, "running", False):
            append_log("Detector already running")
            return
        # persist thresholds to config
        try:
            config["ear_thresh"] = float(self.slider_ear.get())
            config["mar_thresh"] = float(self.slider_mar.get())
        except:
            pass
        save_config(config)
        # find camera
        cam_idx = None
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            except:
                cap = cv2.VideoCapture(i)
            ok, _ = cap.read()
            cap.release()
            if ok:
                cam_idx = i
                break
        if cam_idx is None:
            messagebox.showerror("Camera", "No camera found. Connect webcam and retry.")
            append_log("No camera found")
            return
        self.detector = Detector(self._ui_update, cam_index=cam_idx)
        try:
            self.detector.start_camera()
            append_log(f"Started detector on camera {cam_idx}")
            # update indicator
            self.lbl_rec.configure(text="REC: ON" if config.get("record") else "REC: OFF")
        except Exception as e:
            append_log("Start error: " + str(e))
            messagebox.showerror("Start error", str(e))
            self.detector = None

    def stop_detection(self):
        if self.detector:
            self.detector.stop()
            append_log("Stopped detector")
            self.detector = None
            self.lbl_rec.configure(text="REC: OFF")
            self.lbl_alarm.configure(text="ALARM: OFF")

    def auto_calibrate(self):
        if not self.detector or not getattr(self.detector, "running", False):
            messagebox.showwarning("Calibrate", "Start detection first")
            return
        append_log("Auto-calibration started (3s)")
        def _do():
            samples = []
            end = time.time() + 3.0
            while time.time() < end:
                if len(self.ear_hist) > 0:
                    samples.append((self.ear_hist[-1], self.mar_hist[-1] if len(self.mar_hist)>0 else 0.0))
                time.sleep(0.05)
            if not samples:
                append_log("Calibration failed: no samples")
                return
            arr = np.array(samples)
            meanE = float(np.mean(arr[:,0]))
            meanM = float(np.mean(arr[:,1]))
            newE = max(0.12, round(meanE * 0.72, 3))
            newM = max(0.45, round(meanM * 1.3, 3))
            config["ear_thresh"] = newE
            config["mar_thresh"] = newM
            save_config(config)
            self.root.after(0, lambda: self.slider_ear.set(newE))
            self.root.after(0, lambda: self.slider_mar.set(newM))
            append_log(f"Calibration complete: EAR={meanE:.3f} MAR={meanM:.3f}")
        threading.Thread(target=_do, daemon=True).start()

    def _ui_update(self, frame, ear, mar, status, fps):
        # called in detector thread
        # store last frame
        try:
            self._last_frame = frame.copy()
        except:
            self._last_frame = None
        # push to histories
        self.ear_hist.append(ear)
        self.mar_hist.append(mar)
        # schedule UI render
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img).resize((820,580))
            if status != "Normal":
                pil = ImageEnhance.Brightness(pil).enhance(1.04)
            imgtk = ImageTk.PhotoImage(pil)
        except Exception:
            return
        self.root.after(1, lambda: self._render_frame(imgtk, ear, mar, status, fps))

    def _render_frame(self, imgtk, ear, mar, status, fps):
        try:
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except:
            pass
        self.lbl_ear.configure(text=f"EAR: {ear:.3f}")
        self.lbl_mar.configure(text=f"MAR: {mar:.3f}")
        self.lbl_fps.configure(text=f"FPS: {fps:.1f}")
        # update neon alarm indicator
        if status in ("DROWSY","YAWN"):
            self.lbl_alarm.configure(text="ALARM: ON")
        elif status == "Recovered":
            self.lbl_alarm.configure(text="ALARM: RECOVERED")
        else:
            self.lbl_alarm.configure(text="ALARM: OFF")
        # draw gauges and LEDs
        self._draw_gauges(ear, mar)
        # draw graph
        self._draw_graph()
        # log if alert
        if status != "Normal":
            ln = append_log(status)
            try:
                self.log_box.configure(state="normal")
                self.log_box.insert("end", ln + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
            except:
                pass
        # ThingSpeak indicator update
        has_key = bool(config.get("thingspeak_key"))
        self.ts_status_label.configure(text=("TS: ON" if has_key else "TS: OFF"), text_color=("#76FF03" if has_key else "#FF5252"))

    def _draw_gauges(self, ear, mar):
        try:
            g = self.gauge
            g.delete("all")
            # EAR
            cx1, cy1, r = 100, 100, 70
            v1 = np.clip((ear - 0.08) / 0.4, 0, 1)
            ang = int(240 * v1)
            color_ear = "#00ff99" if ear >= config.get("ear_thresh", DEFAULTS["ear_thresh"]) else "#ff9933"
            g.create_arc(cx1 - r, cy1 - r, cx1 + r, cy1 + r, start=150, extent=240, style="arc", outline="#0b1216", width=16)
            g.create_arc(cx1 - r, cy1 - r, cx1 + r, cy1 + r, start=150, extent=ang, style="arc", outline=color_ear, width=12)
            g.create_text(cx1, cy1, text=f"EAR\n{ear:.3f}", fill="white")
            # MAR
            cx2, cy2 = 280, 100
            v2 = np.clip((mar - 0.2) / 1.2, 0, 1)
            ang2 = int(240 * v2)
            color_mar = "#ffd700" if mar < config.get("mar_thresh", DEFAULTS["mar_thresh"]) else "#ff4444"
            g.create_arc(cx2 - r, cy2 - r, cx2 + r, cy2 + r, start=150, extent=240, style="arc", outline="#0b1216", width=16)
            g.create_arc(cx2 - r, cy2 - r, cx2 + r, cy2 + r, start=150, extent=ang2, style="arc", outline=color_mar, width=12)
            g.create_text(cx2, cy2, text=f"MAR\n{mar:.3f}", fill="white")
            # LED bar
            L = self.led
            L.delete("all")
            n = 12; pad = 4
            W = 350; H = 36
            ratio = 1.0 - np.clip(ear / (config.get("ear_thresh", DEFAULTS["ear_thresh"]) * 1.2 + 1e-6), 0, 1)
            ledw = (W - (n + 1) * pad) / n
            for i in range(n):
                x0 = pad + i * (ledw + pad)
                y0 = pad
                x1 = x0 + ledw
                y1 = H - pad
                lvl = (i + 1) / n
                if lvl <= ratio:
                    col = "#ff4444" if lvl > 0.7 else ("#ff9933" if lvl > 0.4 else "#00ff66")
                    L.create_rectangle(x0, y0, x1, y1, fill=col, outline="")
                else:
                    L.create_rectangle(x0, y0, x1, y1, fill="#111", outline="")
        except Exception as e:
            append_log("Gauge draw error: " + str(e))

    def _draw_graph(self):
        try:
            G = self.graph_canvas
            G.delete("all")
            if len(self.ear_hist) < 4:
                return
            length = len(self.ear_hist)
            xs = np.linspace(0, 1240, length)
            e = np.array(self.ear_hist)
            m = np.array(self.mar_hist)
            vmin = min(e.min(), m.min())
            vmax = max(e.max(), m.max())
            d = vmax - vmin
            if d <= 0: d = 1.0
            ey = 160 - ((e - vmin) / d * 140)
            my = 160 - ((m - vmin) / d * 140)
            ptsE = []; ptsM = []
            for i in range(length):
                ptsE += [xs[i], float(ey[i])]
                ptsM += [xs[i], float(my[i])]
            for i in range(2):
                G.create_line(0, (i+1)*40, 1240, (i+1)*40, fill="#021")
            G.create_line(ptsE, fill="#00ff66", width=2)
            G.create_line(ptsM, fill="#ffd700", width=2)
        except Exception as e:
            append_log("Graph draw error: " + str(e))

    # ---------------- Utilities / actions ----------------
    def take_snapshot(self):
        if self._last_frame is None:
            messagebox.showwarning("Snapshot", "No frame available yet")
            return
        path = save_snapshot(self._last_frame, "manual")
        messagebox.showinfo("Snapshot", f"Saved: {path}")

    def save_logs_csv(self):
        try:
            dest = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
            if not dest:
                return
            with open(LOG_FILE, "r") as src, open(dest, "w", newline="") as dst:
                dst.write(src.read())
            messagebox.showinfo("Export", f"Logs exported to {dest}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def export_pdf_report(self):
        if not HAS_FPDF:
            messagebox.showwarning("PDF", "FPDF not installed. Install 'fpdf' to enable PDF export.")
            return
        try:
            dest = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files","*.pdf")])
            if not dest:
                return
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="DMS ULTRA Report", ln=1, align='C')
            pdf.cell(200, 8, txt=f"Generated: {now_ts()}", ln=1)
            pdf.ln(5)
            # include last 100 log lines
            if LOG_FILE.exists():
                with open(LOG_FILE, "r") as f:
                    lines = f.readlines()[-100:]
                for ln in lines:
                    pdf.multi_cell(0, 6, ln.strip())
            pdf.output(dest)
            messagebox.showinfo("PDF", f"Report saved: {dest}")
        except Exception as e:
            messagebox.showerror("PDF error", str(e))

    # ---------------- UI tick / housekeeping ----------------
    def _ui_tick(self):
        # indicator updates (recording)
        rec_state = config.get("record", False)
        self.lbl_rec.configure(text="REC: ON" if rec_state else "REC: OFF")
        # update TS status dot color
        ts_key = config.get("thingspeak_key", "")
        self.ts_status_label.configure(text=("TS: ON" if ts_key else "TS: OFF"), text_color=("#76FF03" if ts_key else "#FF5252"))
        self.root.after(1000, self._ui_tick)

    def on_close(self):
        try:
            if self.detector:
                self.detector.stop()
        except:
            pass
        save_config(config)
        self.root.destroy()

# ---------------- Main ----------------
def main():
    root = ctk.CTk()
    app = UltraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()