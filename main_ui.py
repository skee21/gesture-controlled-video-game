import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mediapipe as mp
import keyboard
import time
import os

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
            '3_fingers': ' ' # Space (brake)
        }
        
        self.current_key = None
        
        self.last_key_time = 0
        self.key_cooldown = 0.1  
        
        # UI elements
        self.show_help = True
        self.help_timeout = time.time() + 10  # Show help for 10 seconds initially
    
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
            self.last_key_time = current_time
    
    def release_all_keys(self):
        for key in self.gesture_map.values():
            keyboard.release(key)
        self.current_key = None
    
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

        cv2.putText(panel, "Press 'H' for Help | 'Q' to Quit", (w // 2 - 150, panel_height - 10),
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
            
            cv2.imshow(window_name, display_image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.show_help = not self.show_help
                if self.show_help:
                    self.help_timeout = time.time() + 10  

        self.release_all_keys()
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("Initializing Hand Gesture Controller...")
    controller = HandGestureController()
    
    model_path = 'models/Hypermodel.h5'
    print(f"Loading pre-trained model from {model_path}")
    controller.model = tf.keras.models.load_model(model_path)
    
    print("\nStarting Hand Gesture Game Controller")
    print("\nControls:")
    print("  Open Hand (Palm) → W (forward)")
    print("  Closed Hand (Fist) → S (backward)")
    print("  One finger → D (right)")
    print("  Two fingers → A (left)")
    print("  Five fingers → Space (brake/action)")
    print("\nUI Controls:")
    print("  Press 'H' to toggle help overlay")
    print("  Press 'Q' to quit")
    
    controller.run()

if __name__ == "__main__":
    main()