import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mediapipe as mp
import keyboard
import time

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
            'right': 'd',       # Right
            'left': 'a',        # Left
            '3_fingers': ' ' # Space (brake)
        }
        
        self.current_key = None
        
        self.last_key_time = 0
        self.key_cooldown = 0.1  
    
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
    
    def detect_gesture(self, hand_img):
        import random
        return random.choice(list(self.gesture_map.keys()))
    
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
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture frame from camera")
                break
            
            image = cv2.flip(image, 1)
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    gesture = self.process_landmarks(image, hand_landmarks)
                    
                    self.press_key(gesture)
                    
                    cv2.putText(
                        image, 
                        f"Gesture: {gesture} (Key: {self.gesture_map[gesture]})", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
            else:
                self.release_all_keys()
            
            cv2.imshow('Hand Gesture Controller', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        self.release_all_keys()
        cap.release()
        cv2.destroyAllWindows()

def main():
    controller = HandGestureController()
    controller.model = tf.keras.models.load_model('models/Hypermodel.h5')
    
    print("Starting Hand Gesture Controller")
    print("Controls:")
    print("  Palm (open hand) → W (forward)")
    print("  Fist (closed hand) → S (backward)")
    print("  One finger → D (right)")
    print("  Two finger → A (left)")
    print("  Three fingers → Space")
    print("Press 'q' to quit")
    
    controller.run()

if __name__ == "__main__":
    main()