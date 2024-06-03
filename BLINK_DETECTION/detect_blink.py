import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import imutils

# Constants for EAR and blink detection
EYE_AR_THRESH = 0.22 
EYE_AR_CONSEC_FRAMES = 3

# Load the dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize blink counters
left_counter = 0  
right_counter = 0  

# Define facial landmark indices for eyes
LEFT_EYE_POINTS = list(range(36, 42))  
RIGHT_EYE_POINTS = list(range(42, 48))  

# Function to compute the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  
    ear = (A + B) / (2.0 * C)  
    return ear 

# Start video stream
video_capture = cv2.VideoCapture(0)  

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    rects = detector(gray, 0)
    frame = imutils.resize(frame, width=600)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = shape[LEFT_EYE_POINTS]  
        right_eye = shape[RIGHT_EYE_POINTS]  
        left_ear = eye_aspect_ratio(left_eye)  
        right_ear = eye_aspect_ratio(right_eye)

        # Check if the eye aspect ratio is below the blink threshold
        if left_ear < EYE_AR_THRESH:
            left_counter += 1
        else:
            if left_counter >= EYE_AR_CONSEC_FRAMES:
                print("Left eye blinked") 
            left_counter = 0

        if right_ear < EYE_AR_THRESH:  
            right_counter += 1  
        else:
            if right_counter >= EYE_AR_CONSEC_FRAMES: 
                print("Right eye blinked")  
            right_counter = 0
    
    # Display the frame
    cv2.imshow("Blink Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
