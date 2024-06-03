import cv2
import numpy as np
import config as cfg
import f_utils
import dlib
import time

def detect(img, cascade):
    rects, _, confidence = cascade.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True)
    if len(rects) == 0:
        return (), ()
    rects[:, 2:] += rects[:, :2]
    return rects, confidence

def convert_rightbox(img, box_right):
    res = np.array([])
    _, x_max = img.shape
    for box_ in box_right:
        box = np.copy(box_)
        box[0] = x_max - box_[2]
        box[2] = x_max - box_[0]
        if res.size == 0:
            res = np.expand_dims(box, axis=0)
        else:
            res = np.vstack((res, box))
    return res

class DetectFaceOrientation:
    def __init__(self):
        # Create the frontal face detector
        self.detect_frontal_face = cv2.CascadeClassifier(cfg.detect_frontal_face)
        # Create the profile face detector
        self.detect_perfil_face = cv2.CascadeClassifier(cfg.detect_perfil_face)

    def face_orientation(self, gray):
        # Left face
        box_left, w_left = detect(gray, self.detect_perfil_face)
        if len(box_left) == 0:
            box_left = []
            name_left = []
        else:
            name_left = len(box_left) * ["right"]
        
        # Right face
        gray_flipped = cv2.flip(gray, 1)
        box_right, w_right = detect(gray_flipped, self.detect_perfil_face)
        if len(box_right) == 0:
            box_right = []
            name_right = []
        else:
            box_right = convert_rightbox(gray, box_right)
            name_right = len(box_right) * ["left"]

        boxes = list(box_left) + list(box_right)
        names = list(name_left) + list(name_right)
        if len(boxes) == 0:
            return boxes, names
        else:
            index = np.argmax(f_utils.get_areas(boxes))
            boxes = [boxes[index].tolist()]
            names = [names[index]]
        return boxes, names

def draw_bounding_box(img, box, color=(255, 0, 0), thickness=2):
    if len(box) > 0:
        (x1, y1, x2, y2) = box[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def main():
    tpas = 0
    tfas = 0
    cap = cv2.VideoCapture(0)
    face_orientation_detector = DetectFaceOrientation()
    
    def ask_action(action):
        print(f"Please {action}.")
        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"{action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Action Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def detect_movement(expected_action):
        text_pass = 0
        text_fail = 0
        ret, frame = cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, names = face_orientation_detector.face_orientation(gray)
        result_text = "No face detected"
        if len(boxes) > 0:
            draw_bounding_box(frame, boxes)
            detected_action = names[0]
            if detected_action == expected_action:
                result_text = "Test Passed"
                text_pass += 1
            else:
                result_text = "Test Failed"
                text_fail += 1
        
        cv2.putText(frame, result_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Action Detector", frame)
        cv2.waitKey(2000)

        return text_fail, text_pass


    def wait_for_face_detection():
        print("Waiting for face detection...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_orientation_detector.detect_frontal_face.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                print("Face detected. Starting the test...")
                break
            cv2.putText(frame, "Please align your face with the camera", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    wait_for_face_detection()

    actions_to_perform = ["left", "right"]
    for action in actions_to_perform:
        ask_action(action)
        p,f = detect_movement(action)
        tpas += p
        tfas += f

    cap.release()
    cv2.destroyAllWindows()

    return tfas, tpas

if __name__ == "__main__":
    count_test_passed, count_test_failed = main()
    print(f"Test passed: {count_test_passed}")
    print(f"Test failed: {count_test_failed}")
