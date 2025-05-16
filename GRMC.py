import threading
import tkinter as tk
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv
import queue
import pyautogui
import time
import os
import sys

# Global variables
class AppState:
    def __init__(self):
        self.running = False
        self.cap = None
        self.recognizer = None
        self.root = None
        self.start_button = None
        self.stop_button = None
        self.command_queue = queue.Queue()
        self.previous_gesture = None  # Add this line to track previous gesture
        self.last_volume_action = 0  # Add timestamp for volume control cooldown
        self.volume_cooldown = 0.2  # Cooldown in seconds
        self.gesture_start_time = 0  # Track when gesture started
        self.hold_duration = 1.5   # Duration in seconds to hold gesture
        self.current_hold_gesture = None  # Track currently held gesture
        self.last_swap_action = 0  # Add this missing variable
        self.swap_video_cooldown = 0.7  # Add cooldown for next/previous
        self.show_live = None


app = AppState()

def get_model_path():
    """Get correct model path for both development and executable"""
    if getattr(sys, 'frozen', False):
        # Running as executable
        return os.path.join(sys._MEIPASS, 'gesture_recognizer.task')
    else:
        # Running in development
        return 'D:\\AI project\\gesture_recognizer.task'


def start():
    """Handle start button click"""
    if not app.running:
        app.running = True
        app.start_button.config(state=tk.DISABLED)
        app.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=run_recognition, daemon=True).start()

def stop():
    """Handle stop button click"""
    if app.running:
        app.running = False
        # Queue the cleanup command
        app.command_queue.put("cleanup")

def create_gui():
    """Create and setup GUI"""
    app.root = tk.Tk()
    app.show_live = tk.BooleanVar(value=False)
    app.root.title("Gesture Recognition Media Control")
    
    # Set window size and position
    window_width = 400
    window_height = 230  # Increased height for checkbox
    
    # Get screen dimensions
    screen_width = app.root.winfo_screenwidth()
    screen_height = app.root.winfo_screenheight()
    
    # Calculate position for center of screen
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    # Set window size and position
    app.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # Create main frame
    main_frame = tk.Frame(app.root)
    main_frame.pack(expand=True, fill='both', padx=20, pady=20)
    

    
    # Buttons
    app.start_button = tk.Button(main_frame, text="Start", 
                                command=start, width=10, height=2)
    app.start_button.pack(pady=10)
    
    app.stop_button = tk.Button(main_frame, text="Stop", 
                               command=stop, width=10, height=2, 
                               state=tk.DISABLED)
    app.stop_button.pack(pady=10)
        # Add checkbox for show_live
    show_live_cb = tk.Checkbutton(main_frame, 
                                 text="Show Camera Feed", 
                                 variable=app.show_live)
    show_live_cb.pack(pady=10)
    app.root.protocol("WM_DELETE_WINDOW", on_closing)
    # Start GUI update loop
    update_gui()
    app.root.mainloop()

def update_gui():
    """Update GUI and process commands"""
    try:
        # Check for commands in queue
        while True:
            cmd = app.command_queue.get_nowait()
            if cmd == "cleanup_complete":
                app.start_button.config(state=tk.NORMAL)
                app.stop_button.config(state=tk.DISABLED)
    except queue.Empty:
        pass
    
    # Schedule next update
    if app.root and app.root.winfo_exists():
        app.root.after(100, update_gui)

def on_closing():
    """Handle window closing"""
    app.running = False
    if app.root:
        app.root.destroy()

def cleanup_resources():
    """Clean up camera and recognizer resources"""
    if app.cap is not None:
        app.cap.release()
        app.cap = None
    
    if app.recognizer is not None:
        app.recognizer.close()
        app.recognizer = None
    
    cv.destroyAllWindows()

def run_recognition():
    """Main recognition loop"""
    try:
        # Initialize camera
        app.cap = cv.VideoCapture(0)
        print("Initializing camera...")
        cv.waitKey(1000)  # Wait for camera to initialize
        
        app.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        app.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not app.cap.isOpened():
            print("Cannot open camera")
            return

        print("Camera initialized successfully")

        # Initialize recognizer
        print("Initializing gesture recognizer...")
        try:
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                min_hand_detection_confidence = 0.8,
                result_callback=print_result)
            
            app.recognizer = GestureRecognizer.create_from_options(options)
            print("Gesture recognizer initialized successfully")
        except Exception as e:
            print(f"Failed to initialize recognizer: {e}")
            return

        print("Starting recognition loop...")
        while app.running:
            try:
                ret, frame = app.cap.read()
                if not ret or frame is None:
                    if app.running:
                        print("Failed to grab frame")
                    continue

                # Handle camera window
                if app.show_live.get():
                    cv.imshow("Camera", frame)
                    key = cv.waitKey(1)
                    if key == ord('q'):
                        break
                else:
                    # Try to close window if it exists
                    try:
                        cv.destroyWindow("Camera")
                    except:
                        pass

                # Process frame
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Perform recognition
                timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)
                app.recognizer.recognize_async(mp_image, timestamp_ms)

            except Exception as e:
                print(f"Error in recognition loop: {e}")
                if not app.running:
                    break
                continue

    except Exception as e:
        print(f"Critical error in recognition: {e}")
    finally:
        cleanup_resources()
        app.command_queue.put("cleanup_complete")
# Mediapipe setup
# model_path = 'D:\\AI project\\gesture_recognizer.task'

model_path = get_model_path()
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if app.running and result.gestures:
        current_gesture = result.gestures[0][0].category_name
        current_time = time.time()  # Get current time
        
        # Check for gesture change from Open_Palm to Closed_Fist
        if (app.previous_gesture == "Open_Palm" and 
            current_gesture == "Closed_Fist"):
            pyautogui.press('playpause')
            print("Media Play/Pause triggered")
            
        # Volume controls with cooldown
        if current_time - app.last_volume_action > app.volume_cooldown:
            if current_gesture == "Thumb_Up":
                pyautogui.press('volumeup')
                print("Volume Up")
                app.last_volume_action = current_time
            elif current_gesture == "Thumb_Down":
                pyautogui.press('volumedown')
                print("Volume Down")
                app.last_volume_action = current_time
        # Handle next/previous with gesture holding
        if current_time - app.last_swap_action > app.swap_video_cooldown:
            # Start timing if gesture begins
            if current_gesture in ["Pointing_Up", "Closed_Fist"]:
                if app.current_hold_gesture != current_gesture:
                    app.current_hold_gesture = current_gesture
                    app.gesture_start_time = current_time
                else:
                    # Check if gesture has been held long enough
                    if current_time - app.gesture_start_time >= app.hold_duration:
                        if current_gesture == "Pointing_Up":
                            pyautogui.press('nexttrack')
                            print("Next Video")
                            app.last_swap_action = current_time
                        elif current_gesture == "Closed_Fist":
                            pyautogui.press('prevtrack')
                            print("Previous Video")
                            app.last_swap_action = current_time
                        # Reset the hold timer
                        app.gesture_start_time = current_time
            else:
                # Reset if gesture changes to something else
                app.current_hold_gesture = None
                app.gesture_start_time = 0
        # Update previous gesture
        if current_gesture != "None":
            app.previous_gesture = current_gesture
        
        # Print current gesture for debugging
        # print(f'Current gesture: {current_gesture}')

# Start the application
if __name__ == '__main__':
    create_gui()