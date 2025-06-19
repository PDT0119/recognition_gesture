import threading
import customtkinter as ctk
import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2 as cv
import queue
import pyautogui
import time
import os
import sys
from PIL import Image

class AppState:
    def __init__(self):
        self.running = False
        self.cap = None
        self.recognizer = None
        self.command_queue = queue.Queue()
        self.previous_gesture = None
        self.last_volume_action = 0
        self.volume_cooldown = 0.2
        self.gesture_start_time = 0
        self.hold_duration = 1.5
        self.current_hold_gesture = None
        self.last_swap_action = 0
        self.swap_video_cooldown = 0.7
        self.show_live = None
        self.video_label = None

def get_model_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'gesture_recognizer.task')
    else:
        return os.path.join(os.path.dirname(__file__), 'gesture_recognizer.task')

class GestureUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.app_state = AppState()
        
        # Window setup
        self.title("Gesture Recognition Control")
        self.geometry("1280x720")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Create UI elements
        self.create_header()
        self.create_sidebar()
        self.create_main_content()

        # Initialize MediaPipe
        self.setup_mediapipe()
        
        # Set up close handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start GUI update loop
        self.update_gui()

    def create_header(self):
        self.header = ctk.CTkFrame(self, height=80)
        self.header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        self.title_label = ctk.CTkLabel(
            self.header, 
            text="Gesture Recognition Dashboard",
            font=("Segoe UI", 24, "bold")
        )
        self.title_label.pack(pady=20)

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200)
        self.sidebar.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self.status_label = ctk.CTkLabel(
            self.sidebar, 
            text="Status: Stopped",
            font=("Segoe UI", 16)
        )
        self.status_label.pack(pady=20)

        self.start_button = ctk.CTkButton(
            self.sidebar,
            text="Start",
            command=self.start_recognition
        )
        self.start_button.pack(pady=10, padx=20)

        self.stop_button = ctk.CTkButton(
            self.sidebar,
            text="Stop",
            command=self.stop_recognition,
            state="disabled"
        )
        self.stop_button.pack(pady=10, padx=20)

        self.app_state.show_live = ctk.BooleanVar(value=True)
        self.show_live_check = ctk.CTkCheckBox(
            self.sidebar,
            text="Show Camera",
            variable=self.app_state.show_live
        )
        self.show_live_check.pack(pady=20)

    def create_main_content(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=5)

        self.app_state.video_label = ctk.CTkLabel(
            self.main_frame,
            text="Camera Off",
            font=("Segoe UI", 20)
        )
        self.app_state.video_label.pack(expand=True)

    def setup_mediapipe(self):
        try:
            model_path = get_model_path()
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                min_hand_detection_confidence=0.8,
                min_tracking_confidence=0.8,
                min_hand_presence_confidence=0.8,
                result_callback=self.process_result
            )
            self.app_state.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)
            print("MediaPipe gesture recognizer initialized successfully")
        except Exception as e:
            print(f"Error setting up MediaPipe: {e}")
            raise

    def process_result(self, result, output_image, timestamp_ms):
        if result.gestures and self.app_state.running:
            try:
                current_gesture = result.gestures[0][0].category_name
                current_time = time.time()

                # Print for debugging
                print(f"Current gesture: {current_gesture}, Previous: {self.app_state.previous_gesture}")

                # Handle play/pause
                if (self.app_state.previous_gesture == "Open_Palm" and 
                    current_gesture == "Closed_Fist"):
                    print("Play/Pause triggered")
                    pyautogui.press('playpause')
                    time.sleep(0.1)

                # Handle volume control with cooldown
                if current_time - self.app_state.last_volume_action > self.app_state.volume_cooldown:
                    if current_gesture == "Thumb_Up":
                        pyautogui.press('volumeup')
                        self.app_state.last_volume_action = current_time
                    elif current_gesture == "Thumb_Down":
                        pyautogui.press('volumedown')
                        self.app_state.last_volume_action = current_time


                if current_time - self.app_state.last_swap_action > self.app_state.swap_video_cooldown:
                    # Start timing if gesture begins
                    if current_gesture in ["Pointing_Up", "Closed_Fist"]:
                        if self.app_state.current_hold_gesture != current_gesture:
                            self.app_state.current_hold_gesture = current_gesture
                            self.app_state.gesture_start_time = current_time
                        else:
                            # Check if gesture has been held long enough
                            if current_time - self.app_state.gesture_start_time >= self.app_state.hold_duration:
                                if current_gesture == "Pointing_Up":
                                    pyautogui.press('nexttrack')
                                    print("Next Video")
                                    self.app_state.last_swap_action = current_time
                                elif current_gesture == "Closed_Fist":
                                    pyautogui.press('prevtrack')
                                    print("Previous Video")
                                    self.app_state.last_swap_action = current_time
                                # Reset the hold timer
                                self.app_state.gesture_start_time = current_time
                    else:
                        # Reset if gesture changes to something else
                        self.app_state.current_hold_gesture = None
                        self.app_state.gesture_start_time = 0


                # Update previous gesture
                if current_gesture != "None":
                    self.app_state.previous_gesture = current_gesture

            except Exception as e:
                print(f"Error processing gesture: {e}")

    def start_recognition(self):
        try:
            # Reinitialize MediaPipe if needed
            if self.app_state.recognizer is None:
                self.setup_mediapipe()
            
            self.app_state.running = True
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status_label.configure(text="Status: Running")
            threading.Thread(target=self.recognition_loop, daemon=True).start()
        except Exception as e:
            print(f"Error starting recognition: {e}")
            self.status_label.configure(text="Status: Error")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def stop_recognition(self):
        try:
            self.app_state.running = False
            self.status_label.configure(text="Status: Stopping...")
            
            # Clean up resources
            self.cleanup_resources()
            
            # Update UI and restore "Camera Off" text
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_label.configure(text="Status: Stopped")
            
            # Clear video and show "Camera Off" text
            self.app_state.video_label.configure(
                image=None, 
                text="Camera Off",
                font=("Segoe UI", 20)
            )
        except Exception as e:
            print(f"Error stopping recognition: {e}")

    def recognition_loop(self):
        self.app_state.cap = cv.VideoCapture(0)
        
        while self.app_state.running:
            ret, frame = self.app_state.cap.read()
            if not ret:
                continue

            # Process frame
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Update UI if show live is enabled
            if self.app_state.show_live.get():
                self.update_video_feed(frame)

            # Process gesture
            timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)
            self.app_state.recognizer.recognize_async(mp_image, timestamp_ms)

    def update_video_feed(self, frame):
        try:
            frame = cv.resize(frame, (640, 480))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            photo = ctk.CTkImage(
                light_image=Image.fromarray(frame),
                size=(640, 480)
            )
            # Clear the text before showing video
            self.app_state.video_label.configure(image=photo, text="")
            self.app_state.video_label.image = photo
        except Exception as e:
            print(f"Error updating video feed: {e}")

    def cleanup_resources(self):
        try:
            if self.app_state.cap is not None:
                self.app_state.cap.release()
                self.app_state.cap = None
            
            if self.app_state.recognizer is not None:
                self.app_state.recognizer.close()
                self.app_state.recognizer = None
                
            cv.destroyAllWindows()
        except Exception as e:
            print(f"Error cleaning up resources: {e}")

    def update_gui(self):
        if self.winfo_exists():
            self.after(100, self.update_gui)

    def on_closing(self):
        self.app_state.running = False
        self.cleanup_resources()
        self.destroy()

if __name__ == "__main__":
    app = GestureUI()
    app.mainloop()