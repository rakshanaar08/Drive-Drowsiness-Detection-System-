# INSTALL:  pip install opencv-python mediapipe numpy scipy pygame
# RUN:      python drowsiness_detector.py
# KEYS:     Q = quit   |   C = recalibrate EAR
# NOTE:     On first launch, sit in normal driving posture, keep eyes 
#           open during the 5-second calibration countdown.

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import time
import threading
import tkinter as tk
from tkinter import ttk
import pygame
import wave
import struct
import os
import argparse
import platform
from collections import deque
from scipy.spatial import distance as dist
from typing import List, Tuple, Optional

# Conditional Windows imports
if platform.system() == "Windows":
    try:
        import winsound
    except ImportError:
        winsound = None
else:
    winsound = None

# --- CONFIGURATION & SHARED STATE ---

class Config:
    def __init__(self):
        self.lock = threading.Lock()
        
        # EAR Parameters
        self.calibration_ratio = 0.80
        self.ear_consec_seconds = 1.0
        
        # PERCLOS Parameters
        self.perclos_threshold = 0.20
        self.perclos_window_seconds = 60
        
        # MAR Parameters
        self.mar_threshold = 0.50
        self.mar_consec_seconds = 1.0
        
        # Head Pose Parameters
        self.nod_pitch_amplitude_deg = 15.0
        self.pitch_alert_deg = 30.0
        self.roll_alert_deg = 20.0
        
        # Liveness / Anti-Spoofing
        self.liveness_timeout_seconds = 20.0
        
        # Alert Parameters
        self.alarm_repeat_interval = 2.0
        self.mute_alarm = False
        self.recalibrate_flag = False
        
        # Live Metrics (Read-only for UI)
        self.current_ear = 0.0
        self.current_perclos = 0.0
        self.current_mar = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.calibration_status = "INITIALIZING"
        self.ear_open_ref = 0.0

    def update(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get(self, key):
        with self.lock:
            return getattr(self, key)

config = Config()

# --- UTILS ---

def generate_beep(filename="beep.wav", frequency=440, duration=0.5):
    """Generates a synthetic beep tone using the wave module."""
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    
    if os.path.exists(filename):
        return

    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(num_samples):
            value = int(32767.0 * 0.5 * np.sin(2.0 * np.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            f.writeframesraw(data)

# --- MODULES ---

class ConfigPanel(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.root = None

    def run(self):
        self.root = tk.Tk()
        self.root.title("Drowsiness Detector — Config")
        self.root.geometry("400x650")
        
        # Styling
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 10))
        
        # Sliders
        self._create_slider("Calibration Ratio", "calibration_ratio", 0.50, 0.90, 0.01)
        self._create_slider("EAR Consec Seconds", "ear_consec_seconds", 0.3, 3.0, 0.1)
        self._create_slider("PERCLOS Threshold", "perclos_threshold", 0.05, 0.50, 0.01)
        self._create_slider("PERCLOS Window (s)", "perclos_window_seconds", 10, 120, 5)
        self._create_slider("MAR Threshold", "mar_threshold", 0.30, 0.90, 0.01)
        self._create_slider("MAR Consec Seconds", "mar_consec_seconds", 0.3, 3.0, 0.1)
        self._create_slider("Nod Pitch Amp (deg)", "nod_pitch_amplitude_deg", 5, 40, 1)
        self._create_slider("Pitch Alert (deg)", "pitch_alert_deg", 5, 60, 1)
        self._create_slider("Roll Alert (deg)", "roll_alert_deg", 5, 60, 1)
        self._create_slider("Liveness Timeout (s)", "liveness_timeout_seconds", 5, 300, 5)
        self._create_slider("Alarm Interval (s)", "alarm_repeat_interval", 1.0, 10.0, 0.5)

        # Mute Checkbox
        self.mute_var = tk.BooleanVar(value=config.get("mute_alarm"))
        tk.Checkbutton(self.root, text="Mute Alarm", variable=self.mute_var, 
                       command=lambda: config.update(mute_alarm=self.mute_var.get())).pack(pady=10)

        # Recalibrate Button
        tk.Button(self.root, text="Re-Calibrate EAR (C)", bg="#ffcc00", 
                  command=lambda: config.update(recalibrate_flag=True)).pack(pady=10)

        # Metrics display
        self.metrics_label = tk.Label(self.root, text="Live Metrics", font=("Arial", 11, "bold"), fg="#333")
        self.metrics_label.pack(pady=5)
        
        self.stats_text = tk.StringVar()
        tk.Label(self.root, textvariable=self.stats_text, justify=tk.LEFT, font=("Consolas", 10)).pack(pady=5)

        self._update_stats()
        self.root.mainloop()

    def _create_slider(self, label, param_name, min_val, max_val, step):
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame, text=label).pack(side=tk.LEFT)
        
        val = config.get(param_name)
        slider = tk.Scale(frame, from_=min_val, to=max_val, resolution=step, orient=tk.HORIZONTAL,
                          command=lambda v: config.update(**{param_name: float(v)}))
        slider.set(val)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _update_stats(self):
        if self.root:
            stats = (
                f"EAR: {config.get('current_ear'):.3f}\n"
                f"PERCLOS: {config.get('current_perclos'):.1%}\n"
                f"MAR: {config.get('current_mar'):.3f}\n"
                f"Pitch: {config.get('current_pitch'):.1f}°\n"
                f"Yaw: {config.get('current_yaw'):.1f}°\n"
                f"Roll: {config.get('current_roll'):.1f}°"
            )
            self.stats_text.set(stats)
            self.root.after(200, self._update_stats)

class FaceLandmarkExtractor:
    """Wraps MediaPipe FaceMesh to return landmark arrays."""
    def __init__(self, model_path='face_landmarker.task'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Run download script first.")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract(self, frame) -> Optional[np.ndarray]:
        """Returns normalized (x,y,z) coordinates or None."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            # Convert to numpy array
            return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
        return None

class EARCalculator:
    """Computes Eye Aspect Ratio with adaptive calibration."""
    # Landmarks based on MediaPipe FaceMesh
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    @staticmethod
    def calculate_ear(landmarks: np.ndarray, h: int, w: int) -> float:
        """Computes EAR for both eyes and averages them."""
        def eye_aspect_ratio(eye_indices):
            # Convert normalized to pixel coordinates
            pts = np.array([(landmarks[i][0] * w, landmarks[i][1] * h) for i in eye_indices])
            # Vertical pairs
            A = dist.euclidean(pts[1], pts[5])
            B = dist.euclidean(pts[2], pts[4])
            # Horizontal pair
            C = dist.euclidean(pts[0], pts[3])
            return (A + B) / (2.0 * C)

        left_ear = eye_aspect_ratio(EARCalculator.LEFT_EYE)
        right_ear = eye_aspect_ratio(EARCalculator.RIGHT_EYE)
        return (left_ear + right_ear) / 2.0

class PERCLOSTracker:
    """Tracks percentage of time eyes are closed over a rolling window."""
    def __init__(self, window_seconds=60, fps=30):
        self.window_seconds = window_seconds
        self.fps = fps
        self.history = deque(maxlen=window_seconds * fps)

    def update(self, is_closed: bool, window_seconds: int, fps: int) -> float:
        # Check if window/fps changed
        new_maxlen = int(window_seconds * fps)
        if self.history.maxlen != new_maxlen:
            self.history = deque(self.history, maxlen=new_maxlen)
            
        self.history.append(is_closed)
        if not self.history:
            return 0.0
        return self.history.count(True) / len(self.history)

class MARCalculator:
    """Computes Mouth Aspect Ratio for yawn detection."""
    # Vertical indices (Top, Bottom)
    VERTICAL_PAIRS = [(39, 181), (0, 17), (269, 405)]
    # Horizontal indices (Left Corner, Right Corner)
    HORIZONTAL_PAIR = (61, 291)

    @staticmethod
    def calculate_mar(landmarks: np.ndarray, h: int, w: int) -> float:
        """Computes MAR focused on vertical enlargement."""
        pts = lambda idx: (landmarks[idx][0] * w, landmarks[idx][1] * h)
        
        # Vertical lip distances
        v_dists = [dist.euclidean(pts(p[0]), pts(p[1])) for p in MARCalculator.VERTICAL_PAIRS]
        
        # Use Eye-to-Eye distance as a stable reference for scaling
        # (landmarks 33 and 263 are outer corners of eyes)
        eye_ref_dist = dist.euclidean(pts(33), pts(263))
        
        # If eye_ref_dist is 0 (shouldn't happen), avoid div by zero
        if eye_ref_dist < 1e-6: return 0.0
        
        # Return ratio of vertical opening to stable eye distance
        return sum(v_dists) / (2.0 * eye_ref_dist)

class HeadPoseEstimator:
    """Estimates Euler angles and detects nodding."""
    def __init__(self, frame_w, frame_h):
        # 3D model points (standard generic face)
        self.model_pts = np.array([
            (0.0, 0.0, 0.0),          # Nose tip (1)
            (0.0, -330.0, -65.0),    # Chin (152)
            (-225.0, 170.0, -135.0), # Left eye outer corner (33)
            (225.0, 170.0, -135.0),  # Right eye outer corner (263)
            (-150.0, -150.0, -125.0),# Left mouth corner (61)
            (150.0, -150.0, -125.0)  # Right mouth corner (291)
        ], dtype=np.float32)

        # Landmark indices for the model points
        self.landmark_indices = [1, 152, 33, 263, 61, 291]
        
        focal_length = frame_w
        center = (frame_w / 2, frame_h / 2)
        self.cam_matrix = np.array([[focal_length, 0, center[0]],
                                    [0, focal_length, center[1]],
                                    [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
        
        # Nodding detection
        self.pitch_history = deque(maxlen=90) # ~3s at 30fps

    def estimate(self, landmarks, h, w):
        # Extract 2D points
        image_pts = np.array([
            (landmarks[idx][0] * w, landmarks[idx][1] * h) for idx in self.landmark_indices
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(self.model_pts, image_pts, self.cam_matrix, 
                                           self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success:
            return 0, 0, 0, None, None

        # Rodrigues to Matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Decomposition
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        if sy > 1e-6:
            pitch = np.arctan2(rmat[2,1], rmat[2,2])
            yaw = np.arctan2(-rmat[2,0], sy)
            roll = np.arctan2(rmat[1,0], rmat[0,0])
        else:
            pitch = np.arctan2(-rmat[1,2], rmat[1,1])
            yaw = np.arctan2(-rmat[2,0], sy)
            roll = 0

        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        # Correct for coordinate system if necessary; adjust pitch offset
        pitch -= 10 # Offset adjustment for neutral posture
        
        self.pitch_history.append(pitch)
        
        return pitch, yaw, roll, rvec, tvec

    def detect_nodding(self, amp_thresh):
        if len(self.pitch_history) < 30: return False
        
        amp = max(self.pitch_history) - min(self.pitch_history)
        if amp < amp_thresh: return False
        
        # Count zero-crossings (reversals) in derivative
        diffs = np.diff(list(self.pitch_history))
        reversals = 0
        for i in range(len(diffs)-1):
            if diffs[i] * diffs[i+1] < 0:
                reversals += 1
        
        return reversals >= 2

class LivenessTracker:
    """Detects lack of relative facial motion (blinking) to prevent spoofing."""
    def __init__(self):
        self.last_state = None
        self.last_transition_time = 0
        self.still_since = 0

    def update(self, is_ear_closed, curr_time, timeout):
        # Initial state
        if self.last_state is None:
            self.last_state = is_ear_closed
            self.last_transition_time = curr_time
            return 0.0
            
        # Detect transition (blink or open)
        if is_ear_closed != self.last_state:
            self.last_state = is_ear_closed
            self.last_transition_time = curr_time
            return 0.0
        else:
            # How long has it been since the last transition?
            # A real human blinks every few seconds.
            # A photo is perfectly static.
            return curr_time - self.last_transition_time

class AlertManager:
    """Manages visual and audible alerts with hysteresis."""
    def __init__(self, beep_file="beep.wav"):
        pygame.mixer.init()
        try:
            self.beep_sound = pygame.mixer.Sound(beep_file)
        except:
            print("Warning: Pygame audio failed. Using system fallback.")
            self.beep_sound = None
            
        self.alert_active = False
        self.last_trigger_time = 0
        self.last_beep_time = 0
        self.alert_causes = []

    def handle(self, causes, current_time):
        if causes:
            self.alert_active = True
            self.alert_causes = causes
            self.last_trigger_time = current_time
            
            # Audible alert
            repeat_interval = config.get("alarm_repeat_interval")
            if not config.get("mute_alarm"):
                if current_time - self.last_beep_time > repeat_interval:
                    if self.beep_sound:
                        self.beep_sound.play()
                    elif winsound:
                        # Fallback for Windows
                        threading.Thread(target=winsound.Beep, args=(440, 500), daemon=True).start()
                    self.last_beep_time = current_time
        else:
            # Hysteresis: clear alert only after cooldown
            cooldown = 3.0
            if current_time - self.last_trigger_time > cooldown:
                self.alert_active = False
                self.alert_causes = []

    def draw_alert(self, frame):
        if self.alert_active:
            h, w = frame.shape[:2]
            # Red border
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
            
            # Text overlay
            text = "!!! DROWSINESS DETECTED !!!"
            cv2.putText(frame, text, (w//2 - 250, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
            
            cause_text = "Cause: " + " | ".join(self.alert_causes)
            cv2.putText(frame, cause_text, (w//2 - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

class Overlay:
    """Draws metrics and status onto the OpenCV frame."""
    @staticmethod
    def draw(frame, ear, ear_thresh, perclos, mar, mar_thresh, pose, fps, calib_status, ear_open):
        h, w = frame.shape[:2]
        
        def put_text_shadow(img, text, pos, font_scale=0.6, color=(255, 255, 255), thickness=1):
            cv2.putText(img, text, (pos[0]+1, pos[1]+1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+1)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 150), (0, 0, 0), -1)
        cv2.rectangle(overlay, (w-250, 0), (w, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Metrics Top-Left
        ear_col = (0, 255, 0) if ear >= ear_thresh else (0, 0, 255)
        if ear < ear_thresh * 1.2 and ear >= ear_thresh: ear_col = (0, 255, 255)
        
        put_text_shadow(frame, f"EAR: {ear:.3f} (thresh: {ear_thresh:.3f})", (10, 30), color=ear_col)
        
        perclos_col = (0, 255, 0) if perclos < config.get("perclos_threshold") else (0, 0, 255)
        put_text_shadow(frame, f"PERCLOS: {perclos:.1%}", (10, 60), color=perclos_col)
        
        mar_col = (0, 255, 0) if mar < mar_thresh else (0, 0, 255)
        put_text_shadow(frame, f"MAR: {mar:.3f}", (10, 90), color=mar_col)
        
        pitch, yaw, roll = pose
        put_text_shadow(frame, f"P: {pitch:.1f} Y: {yaw:.1f} R: {roll:.1f}", (10, 120))

        # Status Top-Right
        status_col = (0, 255, 0) if calib_status == "DONE" else (0, 165, 255)
        put_text_shadow(frame, f"Status: {calib_status}", (w-240, 30), color=status_col)
        put_text_shadow(frame, f"EAR Open: {ear_open:.3f}", (w-240, 60))

        # Bottom Bar
        put_text_shadow(frame, f"FPS: {int(fps)} | 'C' to Recalibrate | 'Q' to Quit", (10, h-15), font_scale=0.5)

    @staticmethod
    def draw_pose_axis(frame, rvec, tvec, cam_matrix, dist_coeffs):
        if rvec is not None:
            # 3D axis points
            axis_pts = np.array([(50, 0, 0), (0, 50, 0), (0, 0, 50), (0,0,0)], dtype=np.float32)
            imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, cam_matrix, dist_coeffs)
            imgpts = imgpts.astype(int)
            origin = tuple(imgpts[3].ravel())
            cv2.line(frame, origin, tuple(imgpts[0].ravel()), (0, 0, 255), 3) # X - Red
            cv2.line(frame, origin, tuple(imgpts[1].ravel()), (0, 255, 0), 3) # Y - Green
            cv2.line(frame, origin, tuple(imgpts[2].ravel()), (255, 0, 0), 3) # Z - Blue

# --- MAIN ENGINE ---

def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detector")
    parser.add_argument("--no-gui", action="store_true", help="Disable Tkinter config panel")
    args = parser.parse_args()

    # 1. Init Audio
    generate_beep()
    
    # 2. Config Panel
    if not args.no_gui:
        panel = ConfigPanel()
        panel.start()
    else:
        print("Running in NO-GUI mode. Using default thresholds.")
    
    # 3. Video Capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 4. Modules
    extractor = FaceLandmarkExtractor()
    perclos_tracker = PERCLOSTracker()
    pose_estimator = HeadPoseEstimator(640, 480)
    liveness_tracker = LivenessTracker() 
    alert_manager = AlertManager()
    
    # Timing
    prev_time = time.time()
    fps = 30
    
    # EAR Calibration logic
    ear_samples = []
    calibrating = True
    calib_start_time = 0
    ear_open_ref = 0.0
    
    no_face_start = 0
    
    ear_consec_start = 0
    mar_consec_start = 0
    pitch_consec_start = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        curr_time = time.time()
        
        # Calculate Local FPS
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Remote Recalibrate Trigger
        if config.get("recalibrate_flag"):
            calibrating = True
            calib_start_time = 0
            ear_samples = []
            config.update(recalibrate_flag=False, calibration_status="RE-CALIBRATING")

        landmarks = extractor.extract(frame)
        
        if landmarks is not None:
            no_face_start = 0
            
            # --- METRICS CALCULATIONS ---
            
            ear = EARCalculator.calculate_ear(landmarks, h, w)
            mar = MARCalculator.calculate_mar(landmarks, h, w)
            pitch, yaw, roll, rvec, tvec = pose_estimator.estimate(landmarks, h, w)
            
            # Update shared state for UI
            config.update(current_ear=ear, current_mar=mar, 
                          current_pitch=pitch, current_yaw=yaw, current_roll=roll)
            
            # --- CALIBRATION ---
            if calibrating:
                if calib_start_time == 0: calib_start_time = curr_time
                elapsed = curr_time - calib_start_time
                
                if elapsed < 5.0:
                    ear_samples.append(ear)
                    text = f"CALIBRATION: OPEN EYES ({5 - int(elapsed)}s)"
                    cv2.putText(frame, text, (w//2-200, h//2), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                    config.update(calibration_status=f"IN PROGRESS ({int(elapsed)}s)")
                else:
                    ear_open_ref = np.mean(ear_samples)
                    config.update(ear_open_ref=ear_open_ref, calibration_status="DONE")
                    calibrating = False
            
            # --- ALARM LOGIC ---
            causes = []
            ear_threshold = ear_open_ref * config.get("calibration_ratio") if ear_open_ref > 0 else 0.25
            perclos_threshold = config.get("perclos_threshold")
            
            # 1. EAR Persistence
            is_ear_closed = ear < ear_threshold
            if not calibrating:
                if is_ear_closed:
                    if ear_consec_start == 0: ear_consec_start = curr_time
                    if curr_time - ear_consec_start > config.get("ear_consec_seconds"):
                        causes.append(f"EAR Closed {curr_time - ear_consec_start:.1f}s")
                else:
                    ear_consec_start = 0
                
                # 2. PERCLOS
                perclos = perclos_tracker.update(is_ear_closed, 
                                                 config.get("perclos_window_seconds"), int(fps))
                config.update(current_perclos=perclos)
                if perclos > perclos_threshold:
                    causes.append(f"PERCLOS {perclos:.1%}")
                
                # 3. MAR Persistence (Yawn)
                if mar > config.get("mar_threshold"):
                    if mar_consec_start == 0: mar_consec_start = curr_time
                    if curr_time - mar_consec_start > config.get("mar_consec_seconds"):
                        causes.append("Yawning")
                else:
                    mar_consec_start = 0
                    
                # 4. Head Pose (Nodding / Tilt) - GATED by eye indicators
                # We check if eyes are currently closed OR if PERCLOS is high
                if is_ear_closed or perclos > perclos_threshold:
                    if abs(pitch) > config.get("pitch_alert_deg"):
                        if pitch_consec_start == 0: pitch_consec_start = curr_time
                        if curr_time - pitch_consec_start > 1.0:
                            causes.append("Drowsy Pitch (Tilt)")
                    elif abs(roll) > config.get("roll_alert_deg"):
                        if pitch_consec_start == 0: pitch_consec_start = curr_time
                        if curr_time - pitch_consec_start > 1.0:
                            causes.append("Drowsy Roll (Tilt)")
                    else:
                        pitch_consec_start = 0
                    
                    if pose_estimator.detect_nodding(config.get("nod_pitch_amplitude_deg")):
                        causes.append("Nodding Detected")
                else:
                    pitch_consec_start = 0 # Reset pitch if eyes open

                # 5. Liveness Detection (Anti-Spoofing)
                # Detection: If the eye state (Open/Closed) hasn't changed recently.
                # A photo never blinks. A real person blinks every few seconds.
                still_duration = liveness_tracker.update(is_ear_closed, curr_time, 
                                                         config.get("liveness_timeout_seconds"))
                if still_duration > config.get("liveness_timeout_seconds"):
                    causes.append(f"NO BLINK ({int(still_duration)}s)")

            # Handle Alerts
            alert_manager.handle(causes, curr_time)
            
            # --- DRAWING ---
            Overlay.draw(frame, ear, ear_threshold, config.get("current_perclos"), 
                         mar, config.get("mar_threshold"), (pitch, yaw, roll), 
                         fps, config.get("calibration_status"), ear_open_ref)
            
            Overlay.draw_pose_axis(frame, rvec, tvec, pose_estimator.cam_matrix, pose_estimator.dist_coeffs)
            alert_manager.draw_alert(frame)
            
        else:
            # No face detected
            if no_face_start == 0: no_face_start = curr_time
            elapsed_no_face = curr_time - no_face_start
            
            if elapsed_no_face > 3.0: # Alarm after 3 seconds of no face
                alert_manager.handle(["DRIVER MISSING / NO FACE"], curr_time)
                alert_manager.draw_alert(frame)
                cv2.putText(frame, "!!! NO FACE DETECTED !!!", (w//2-180, h//2), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            else:
                alert_manager.handle([], curr_time) # Clear if face returns quickly
            
        cv2.imshow("Driver Drowsiness Monitor", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            config.update(recalibrate_flag=True)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    if os.path.exists("beep.wav"):
        try: os.remove("beep.wav")
        except: pass

if __name__ == "__main__":
    main()
