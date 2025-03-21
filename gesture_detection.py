import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

GESTURE_FILE = "gestures.json"

def save_gesture(gesture_name, data):
    """Saves the gesture landmarks to a JSON file."""
    avg_landmarks = np.mean(data, axis=0).tolist()

    if os.path.exists(GESTURE_FILE):
        with open(GESTURE_FILE, "r") as f:
            gestures = json.load(f)
    else:
        gestures = {}

    gestures[gesture_name] = avg_landmarks

    with open(GESTURE_FILE, "w") as f:
        json.dump(gestures, f, indent=4)

    print(f"Gesture '{gesture_name}' saved successfully!")

def learn_gesture():
    """Captures and saves a new hand gesture."""
    gesture_name = input("Enter the gesture name: ")
    cap = cv2.VideoCapture(0)
    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
                data.append(landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Learning Gesture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and data:
            save_gesture(gesture_name, data)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_gesture():
    """Detects saved gestures in real-time."""
    if not os.path.exists(GESTURE_FILE):
        print("No gestures found! Please learn gestures first.")
        return

    with open(GESTURE_FILE, "r") as f:
        gestures = json.load(f)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]

                detected_gesture = "Unknown"
                min_distance = float('inf')

                for name, saved_landmarks in gestures.items():
                    distance = np.linalg.norm(np.array(saved_landmarks) - np.array(landmarks))
                    if distance < min_distance:
                        min_distance = distance
                        detected_gesture = name

                cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to select learning or detection mode."""
    while True:
        print("\nHand Gesture Detection System")
        print("1: Learn a New Gesture")
        print("2: Detect Gestures in Real Time")
        print("3: Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            learn_gesture()
        elif choice == "2":
            recognize_gesture()
        elif choice == "3":
            break2
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
