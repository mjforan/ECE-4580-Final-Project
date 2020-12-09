import cv2

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('./haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.imshow("face", cv2.resize(frame[y:y + h, x:x + w], (48, 48)))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        eyes = eye_classifier.detectMultiScale(roi_gray)
        if len(eyes) > 2:
            eyes = filter(lambda eye: eye[1] < len(roi_gray) / 2 and min(eye[2], eye[3]) > len(roi_gray) / 4, eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        cv2.putText(frame, "yikes", (x + int((w / 2)), y + 25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Labeled', frame)
    if cv2.waitKey(1) == 13:  # Enter key kills program
        break

cap.release()
cv2.destroyAllWindows()
