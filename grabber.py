import keras
import numpy as np
import cv2
import math

model = keras.models.load_model("./vt-moji-0")
EMOTIONS = {
    0:"angry",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"sad",
    5:"surprise",
    6:"neutral"
}

def predict_image(input_image):
  expanded_image = np.expand_dims(input_image, 0)
  expanded_image = np.expand_dims(expanded_image, -1)
  results = model.predict(expanded_image)[0]
  return EMOTIONS[list(results).index(max(results))]


face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("./haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        eyes = eye_classifier.detectMultiScale(roi_gray)
        if len(eyes) > 2:
            eyes = list(
                filter(lambda eye: eye[1] < len(roi_gray) / 2, eyes))  # and min(eye[2], eye[3]) > len(roi_gray) / 4
        if len(eyes) >= 2:
            sizes = list(map(lambda eye: eye[2] + eye[3], eyes))
            eye0 = eyes[sizes.index(sorted(sizes)[0])]
            eye1 = eyes[sizes.index(sorted(sizes)[1])]
            theta = math.atan2((eye0[1] - eye1[1]), (eye0[0] - eye1[0]))
            if 1 / 4 * math.pi < abs(theta) < 7 / 4 * math.pi:
                theta += math.pi
            rotation = cv2.getRotationMatrix2D((x + eye0[0], y + eye0[1]), theta / math.pi * 180, 1.0)
            rotated = cv2.warpAffine(gray, rotation, gray.shape)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        cv2.putText(frame, predict_image(cv2.resize(rotated[y:y + h, x:x + w], (48,48))), (x + int((w / 2)), y + 25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Labeled', frame)
    if cv2.waitKey(1) == 13:  # Enter key kills program
        break

cap.release()
cv2.destroyAllWindows()
