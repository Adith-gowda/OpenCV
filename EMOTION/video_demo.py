import cv2
import time
from fer import FER

# Initialize the FER detector
detector = FER(mtcnn=True)

# List of emotions to ask the user to display
emotions_to_display = ["happy", "neutral", "angry"]
count_passed = 0
count_failed = 0

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Function to display a question and wait for the user to display the emotion
def ask_emotion(emotion):
    print(f"Please show a {emotion} face.")
    # Display the prompt for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Show a {emotion} face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to detect and display the emotion
def detect_emotion(expected_emotion):
    ret, frame = cap.read()
    if not ret:
        return
    # Detect emotions in the frame
    emotions = detector.detect_emotions(frame)
    result_text = "No face detected"
    if emotions:
        e = emotions[0]
        (x, y, w, h) = e['box']
        detected_emotion = max(e['emotions'], key=e['emotions'].get)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # Check if the detected emotion matches the expected emotion
        if detected_emotion == expected_emotion:
            result_text = "Test Passed"
            count_passed += 1
        else:
            result_text = "Test Failed"
            count_failed += 1
    cv2.putText(frame, result_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Display the frame with detected emotion and result
    cv2.imshow("Emotion Detector", frame)
    cv2.waitKey(2000)  # Wait for 2 seconds to show the detected emotion and result

# Main loop
for emotion in emotions_to_display:
    ask_emotion(emotion)
    detect_emotion(emotion)

# Print the final results
print(f"Test Passed: {count_passed}")
print(f"Test Failed: {count_failed}")


# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
