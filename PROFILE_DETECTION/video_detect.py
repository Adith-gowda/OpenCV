import cv2
import dlib
import time
import numpy as np

# Load the shape predictor model
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(shape_predictor_path)
except RuntimeError:
    print(f"Error: The shape predictor model file '{shape_predictor_path}' was not found.")
    exit(1)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Function to display a question and wait for the user to perform the action
def ask_action(action):
    print(f"Please {action}.")
    # Display the prompt for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"{action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Action Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to detect head movement
def detect_movement(expected_action):
    ret, frame = cap.read()
    if not ret:
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    result_text = "No face detected"
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        
        # Draw bounding box
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw landmarks
        for point in landmarks_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        
        # Calculate the angles to detect head movements
        left_eye = landmarks_points[36]
        right_eye = landmarks_points[45]
        nose_tip = landmarks_points[30]
        chin = landmarks_points[8]

        # Calculate horizontal and vertical angles
        horizontal_angle = calculate_angle(left_eye, nose_tip, right_eye)
        vertical_angle = calculate_angle(nose_tip, chin, (nose_tip[0], chin[1]))

        detected_action = None
        if horizontal_angle > 20:
            detected_action = "turn left" if left_eye[0] < right_eye[0] else "turn right"
        elif vertical_angle > 20:
            detected_action = "look up" if nose_tip[1] < chin[1] else "look down"

        if detected_action == expected_action:
            result_text = "Test Passed"
        else:
            result_text = "Test Failed"
    
    cv2.putText(frame, result_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the frame with detected action and result
    cv2.imshow("Action Detector", frame)
    cv2.waitKey(2000)  # Wait for 2 seconds to show the detected action and result

# Function to wait until a face is detected before starting the prompts
def wait_for_face_detection():
    print("Waiting for face detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            print("Face detected. Starting the test...")
            break
        cv2.putText(frame, "Please align your face with the camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Main loop
wait_for_face_detection()

actions_to_perform = ["turn left", "turn right", "look up", "look down"]
for action in actions_to_perform:
    ask_action(action)
    detect_movement(action)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
