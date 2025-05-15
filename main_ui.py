import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mediapipe as mp
import keyboard
import time
import os
import tkinter as tk
from PIL import Image, ImageTk
import threading
import win32gui
import win32con
import win32api

class OverlayWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Overlay")
        self.root.attributes("-topmost", True)  
        self.root.attributes("-alpha", 0.7)     
        self.root.overrideredirect(True)       
        self.root.wm_attributes("-transparentcolor", "black")  
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.width = 200
        self.height = 150
        self.root.geometry(f"{self.width}x{self.height}+{screen_width-self.width-10}+{screen_height-self.height-60}")
        
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.gesture_label = self.canvas.create_text(self.width//2, 40, text="No gesture", font=("Arial", 14, "bold"), fill="white")
        self.key_label = self.canvas.create_text(self.width//2, self.height//2+20, text="", font=("Arial", 24, "bold"), fill="#00FF00")
        
        if os.name == 'nt':  
            hwnd = win32gui.FindWindow(None, "Gesture Overlay")
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                                  win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | 
                                  win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
        
        close_button = tk.Button(self.root, text="×", command=self.root.destroy, 
                                bg="red", fg="white", bd=0, font=("Arial", 10))
        close_button.place(x=self.width-20, y=0, width=20, height=20)

        self.canvas.bind("<Button-1>", self.start_move)
        self.canvas.bind("<ButtonRelease-1>", self.stop_move)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        
    def start_move(self, event):
        self.x = event.x
        self.y = event.y
        
    def stop_move(self, event):
        self.x = None
        self.y = None
        
    def on_motion(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")
    
    def update_gesture_info(self, gesture_name, key):
        gesture_display = {
            'palm': 'PALM (Forward)',
            'fist': 'FIST (Backward)',
            'right': '1 FINGER (Right)',
            'left': '2 FINGERS (Left)',
            '3_fingers': '3 FINGERS (Brake)',
            None: 'No Gesture'
        }
        
        display_name = gesture_display.get(gesture_name, 'Unknown')
        
        self.canvas.itemconfig(self.gesture_label, text=display_name)
        self.canvas.itemconfig(self.key_label, text=key.upper() if key else "")
        self.root.update()


class HandGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.model = self.create_model()
        
        self.gesture_map = {  
            'palm': 'w',        # Forward
            'fist': 's',        # Backward
            'right': 'd',       # Right (1 finger)
            'left': 'a',        # Left (2 fingers)
            '3_fingers': ' '    # Space (brake)
        }
        
        self.current_key = None
        self.current_gesture = None
        
        self.last_key_time = 0
        self.key_cooldown = 0.1  
        
        # UI elements
        self.show_help = True
        self.help_timeout = time.time() + 10  # Show help for 10 seconds initially

        self.overlay_thread = threading.Thread(target=self.create_overlay)
        self.overlay_thread.daemon = True
        self.overlay = None
        self.overlay_thread.start()
    
    def create_overlay(self):
        self.overlay = OverlayWindow()
        self.overlay.root.mainloop()
    
    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(5, activation='softmax')  
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_hand_image(self, image, hand_landmarks):
        h, w, _ = image.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        hand_img = image[y_min:y_max, x_min:x_max]
        if hand_img.size == 0:
            return None
        
        hand_img = cv2.resize(hand_img, (64, 64))
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        hand_img = hand_img / 255.0
        hand_img = np.reshape(hand_img, (1, 64, 64, 1))
        
        return hand_img
    
    def process_landmarks(self, image, landmarks):
        points = []
        for landmark in landmarks.landmark:
            h, w, _ = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
        
        fingers_extended = []
        for i in range(8, 21, 4):  
            finger_extended = points[i][1] < points[i-2][1]  
            fingers_extended.append(finger_extended)
        
        num_fingers_extended = sum(fingers_extended)
        
        if num_fingers_extended == 1:
            return 'right'  # 1 finger for right
        elif num_fingers_extended == 2:
            return 'left'  # 2 fingers for left
        elif num_fingers_extended == 3: 
            return '3_fingers'  # All fingers extended (space)
        elif num_fingers_extended == 0:
            return 'fist'  # No fingers extended (fist)
        else:
            return 'palm'  # Default to palm (forward)
    
    def press_key(self, gesture):
        current_time = time.time()
  
        if self.current_key and self.current_key != self.gesture_map[gesture]:
            keyboard.release(self.current_key)
            self.current_key = None
        
        if current_time - self.last_key_time > self.key_cooldown:
            key = self.gesture_map[gesture]
            keyboard.press(key)
            self.current_key = key
            self.current_gesture = gesture
            self.last_key_time = current_time

            if self.overlay:
                self.overlay.update_gesture_info(gesture, key)
    
    def release_all_keys(self):
        for key in self.gesture_map.values():
            keyboard.release(key)
        self.current_key = None
        self.current_gesture = None

        if self.overlay:
            self.overlay.update_gesture_info(None, "")
    
    def draw_ui(self, image, detected_gesture=None):
        h, w, _ = image.shape

        panel_height = 100
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel.fill(50)  

        gestures = [
            ('Palm', 'Forward', 'palm'),
            ('Fist', 'Backward', 'fist'),
            ('1 Finger', 'Right', 'right'),
            ('2 Fingers', 'Left', 'left'),
            ('3 Fingers', 'Handbrake', '3_fingers')
        ]

        total = len(gestures)
        spacing = w // total

        for i, (name, key, gid) in enumerate(gestures):
            center_x = spacing * i + spacing // 2
            color = (0, 255, 0) if detected_gesture == gid else (200, 200, 200)

            cv2.putText(panel, name, (center_x - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(panel, key, (center_x - 60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(panel, "Press 'H' for Help | 'Q' to Quit | 'O' for Overlay Only", (w // 2 - 220, panel_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return np.vstack((image, panel))
    
    def draw_help_overlay(self, image):
        h, w, _ = image.shape
   
        overlay = image.copy()
        
        cv2.rectangle(overlay, (50, 50), (w-50, h-50), (30, 30, 30), -1)
        
        help_text = [
            "Hand Gesture Controller - Help",
            "",
            "1. Position your hand in front of the camera",
            "2. Make one of the following gestures:",
            "   - Open hand (palm up) -> Drive forward (W)",
            "   - Closed fist -> Drive backward (S)",
            "   - One finger only -> Turn right (D)",
            "   - Two fingers -> Turn left (A)",
            "   - Three fingers -> Space (Handbrake)",
            "",
            "Tips:",
            "- Keep your hand in the camera's view",
            "- Make clear gestures with good lighting",
            "- Maintain distance from camera (1-2 feet)",
            "- Press 'O' to hide camera and use overlay only",
            "",
            "Press 'H' to hide this help | 'Q' to quit"
        ]
        
        for i, line in enumerate(help_text):
            y = 100 + i * 25
            cv2.putText(overlay, line, (100, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (200, 200, 200), 1, cv2.LINE_AA)
        
        alpha = 0.7
        output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        return output
    
    def run(self):
        window_name = 'Hand Gesture Game Controller'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(0)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.resizeWindow(window_name, width, height + 120)
        
        overlay_only_mode = False
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture frame from camera")
                break
            
            image = cv2.flip(image, 1)
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb_image)
            
            detected_gesture = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    detected_gesture = self.process_landmarks(image, hand_landmarks)
                    
                    self.press_key(detected_gesture)
                    
                    cv2.putText(
                        image, 
                        f"Detected: {detected_gesture.replace('_', ' ').title()} ({self.gesture_map[detected_gesture]})", 
                        (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
            else:
                self.release_all_keys()
                cv2.putText(
                    image,
                    "No hand detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
            
            display_image = self.draw_ui(image, detected_gesture)
            
            if self.show_help:
                if time.time() < self.help_timeout:
                    display_image = self.draw_help_overlay(display_image)
                else:
                    self.show_help = False
            
            if not overlay_only_mode:
                cv2.imshow(window_name, display_image)
            else:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow(window_name)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.show_help = not self.show_help
                if self.show_help:
                    self.help_timeout = time.time() + 10
            elif key == ord('o'):
                overlay_only_mode = not overlay_only_mode
                if not overlay_only_mode:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, width, height + 120)

        self.release_all_keys()
        cap.release()
        cv2.destroyAllWindows()
        
        # Waits for overlay thread to finish
        if self.overlay:
            try:
                self.overlay.root.destroy()
            except:
                pass

def main():
    print("Initializing Hand Gesture Controller...")
    controller = HandGestureController()
    
    model_path = 'models/Hypermodel.h5'
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        controller.model = tf.keras.models.load_model(model_path)
    else:
        print(f"Model file not found at {model_path}. Using default model.")
    
    print("\nStarting Hand Gesture Game Controller")
    print("\nControls:")
    print("  Open Hand (Palm) → W (forward)")
    print("  Closed Hand (Fist) → S (backward)")
    print("  One finger → D (right)")
    print("  Two fingers → A (left)")
    print("  Three fingers → Space (brake/action)")
    print("\nUI Controls:")
    print("  Press 'H' to toggle help overlay")
    print("  Press 'O' to toggle overlay-only mode (hide camera window)")
    print("  Press 'Q' to quit")
    
    controller.run()

if __name__ == "__main__":
    main()