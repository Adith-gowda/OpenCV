import cv2
from fer import FER
import matplotlib.pyplot as plt

# Load the image
image_path = '../images/happy1.jpeg'
image = cv2.imread(image_path)

# Initialize the FER detector
detector = FER(mtcnn=True)

# Detect emotions in the image
emotions = detector.detect_emotions(image)

# Print the detected emotions
print(emotions)

print(f"Box: {emotions[0]['box']}")
print(f"Emotions: {emotions[0]['emotions']}")

# Optionally, display the image with detected emotion
for e in emotions:
    (x, y, w, h) = e['box']
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    emotion_text = max(e['emotions'], key=e['emotions'].get)
    cv2.putText(image, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Convert image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
