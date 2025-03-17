from ultralytics import YOLO
import cv2
import mediapipe as mp

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

# Function to detect hand gesture
def detect_gesture(hand_landmarks):
    # Get landmark positions
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # Threshold untuk menentukan jari terangkat
    threshold = 0.05  # Sesuaikan nilai ini jika diperlukan

    # Check if fingers are raised
    thumb_raised = thumb_tip.y < thumb_ip.y - threshold
    index_raised = index_tip.y < index_pip.y - threshold
    middle_raised = middle_tip.y < middle_pip.y - threshold
    ring_raised = ring_tip.y < ring_pip.y - threshold
    pinky_raised = pinky_tip.y < pinky_pip.y - threshold

    # Determine gesture based on raised fingers
    if thumb_raised and index_raised and not middle_raised and not ring_raised and pinky_raised:
        return "Metal"
    elif thumb_raised and index_raised and middle_raised and ring_raised and pinky_raised:
        return "Jari Terbuka"
    elif not thumb_raised and not index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "Jari Tertutup"
    elif not thumb_raised and index_raised and middle_raised and not ring_raised and not pinky_raised:
        return "Peace"
    elif thumb_raised and not index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "Jempol"
    elif not thumb_raised and index_raised and not middle_raised and not ring_raised and not pinky_raised:
        return "Telunjuk"
    elif not thumb_raised and not index_raised and middle_raised and not ring_raised and not pinky_raised:
        return "Tengah"
    elif not thumb_raised and not index_raised and not middle_raised and ring_raised and not pinky_raised:
        return "Manis"
    elif not thumb_raised and not index_raised and not middle_raised and not ring_raised and pinky_raised:
        return "Kelingking"
    else:
        return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Pose Detection
    results = model(frame)
    for result in results:
        keypoints = result.keypoints
        # Filter keypoints for hands (left_wrist and right_wrist)
        hand_points = []
        for kp in keypoints:
            if kp[0] in [9, 10]:  # Indeks 9: left_wrist, 10: right_wrist
                hand_points.append(kp)
        # Draw hand keypoints
        for point in hand_points:
            x, y = int(point[1]), int(point[2])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # MediaPipe Hands Detection
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(image_rgb)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Detect gesture
            gesture = detect_gesture(hand_landmarks)
            
            # Display gesture text inside the bounding box
            cv2.putText(frame, f"{gesture}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Pose and Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()