import keras
import numpy as np
import cv2
import math

model = keras.models.load_model("./vt-moji-0")
eye_coordinates = np.array([((177, 278), (303, 278)), ((131, 186), (342, 186)), ((150, 231), (331, 231)),
                            ((169, 185), (311, 185)), ((168, 193), (312, 193)), ((164, 192), (312, 192)),
                            ((171, 185), (308, 185))])

emoji = [{"label": label, "image": cv2.imread("./emoji/" + label + ".webp", cv2.IMREAD_UNCHANGED),
          "eyes": eye_coordinates[i]}
         for i, label in enumerate(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"])]


def predict_image(input_image):
    expanded_image = np.expand_dims(input_image, 0)
    expanded_image = np.expand_dims(expanded_image, -1)
    results = model.predict(expanded_image)[0]
    return emoji[list(results).index(max(results))]


face_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("./haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if x < 0 or y < 0 or x + w >= frame.shape[0] or y + h >= frame.shape[1]:
            continue

        roi_gray = gray[y:y + h, x:x + w]
        cv2.imshow("hidden face", frame[y:y+h, x:x+w])
        eyes = eye_classifier.detectMultiScale(roi_gray)
        if len(eyes) > 2:
            eyes = list(filter(lambda eye: eye[1] < len(roi_gray) / 2, eyes))
        if len(eyes) >= 2:
            sizes = list(map(lambda eye: eye[2] + eye[3], eyes))
            eyes = sorted([eyes[sizes.index(sorted(sizes)[0])], eyes[sizes.index(sorted(sizes)[1])]],
                          key=lambda eye: eye[0])
            theta = math.atan2((eyes[0][1] - eyes[1][1]), (eyes[0][0] - eyes[1][0]))
            if 1 / 4 * math.pi < abs(theta) < 7 / 4 * math.pi:
                theta += math.pi
            rotation = cv2.getRotationMatrix2D((x + eyes[0][0], y + eyes[0][1]), theta / math.pi * 180, 1.0)
            rotated = cv2.warpAffine(gray, rotation, gray.shape)

            prediction = predict_image(cv2.resize(rotated[y:y + h, x:x + w], (48, 48)))
            em_img = prediction["image"]

            side = em_img.shape[0]
            rotation = np.append(cv2.getRotationMatrix2D((side / 2, side / 2), -theta / math.pi * 180, 1.0),
                                 np.array([[0, 0, 1]]), axis=0)
            rotated = cv2.warpPerspective(em_img[..., :3], rotation, (side, side))

            scale = np.linalg.norm(eyes[1] - eyes[0]) / np.linalg.norm(prediction["eyes"][1] - prediction["eyes"][0])
            if scale == 0:
                continue
            side = int(em_img.shape[0] * scale)
            new_eye = (np.dot(rotation, np.append(np.array(prediction["eyes"][0]), 1).T) * scale).astype(int)
            em_img_resized = cv2.resize(em_img, (side, side))
            mask = em_img_resized[..., 3:] / 255.0
            x = int(x + eyes[0][0] + eyes[0][2] / 2 - new_eye[0])
            y = int(y + eyes[0][1] + eyes[0][3] / 3 - new_eye[1])
            frame[y:y + side, x:x + side] = (1.0 - mask) * frame[y:y + side, x:x + side] + mask * em_img_resized[...,
                                                                                                  :3]

    cv2.imshow("VT-moji", frame)
    if cv2.waitKey(1) == 13:  # Enter key kills program
        break

cap.release()
cv2.destroyAllWindows()
