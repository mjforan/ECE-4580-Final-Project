import keras
import numpy as np
import cv2
import math

model = keras.models.load_model("./vt-moji-0")

# pixel coordinates of the eye positions for each emoji
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
        roi_gray = gray[y:y + h, x:x + w]
        # display the face which will be hidden behind the emoji
        cv2.imshow("hidden face", frame[y:y+h, x:x+w])
        eyes = eye_classifier.detectMultiScale(roi_gray)
        if len(eyes) > 2:
            # only look for eyes in the top half of the image
            eyes = list(filter(lambda eye: eye[1] < len(roi_gray) / 2, eyes))
        if len(eyes) >= 2:
            # select the two largest eyes and sort them by x-coordinate. The result will be [left_eye, right_eye]
            eyes = sorted(sorted(eyes, key=lambda eye: eye[2]+eye[3])[:3], key=lambda eye: eye[0])
            # rotate the face so that the line connecting the two eyes is horizontal
            theta = math.atan2((eyes[0][1] - eyes[1][1]), (eyes[0][0] - eyes[1][0]))
            if 1 / 4 * math.pi < abs(theta) < 7 / 4 * math.pi:
                theta += math.pi
            rotation = cv2.getRotationMatrix2D((x + eyes[0][0], y + eyes[0][1]), theta / math.pi * 180, 1.0)
            rotated = cv2.warpAffine(gray, rotation, gray.shape)

            # use the loaded model to predict the emoji to use
            try:
                prediction = predict_image(cv2.resize(rotated[y:y + h, x:x + w], (48, 48)))
            except cv2.error:
                # This has something to do with the face going over the right edge of the image, but I'm too tired to figure it out
                continue

            em_img = prediction["image"]

            side = em_img.shape[0]
            # Rotate the emoji image - no need to resize since it's circular
            rotation = np.append(cv2.getRotationMatrix2D((side / 2, side / 2), -theta / math.pi * 180, 1.0), np.array([[0, 0, 1]]), axis=0)
            rotated = cv2.warpPerspective(em_img, rotation, (side, side))

            # Scale the emoji image so the distance between the eyes matches the detected face
            scale = np.linalg.norm(eyes[1] - eyes[0]) / np.linalg.norm(prediction["eyes"][1] - prediction["eyes"][0])
            side = int(em_img.shape[0] * scale)
            # Also apply the rotation to the emoji eye coordinate so it still works for the rotated image
            new_eye = (np.dot(rotation, np.append(np.array(prediction["eyes"][0]), 1).T) * scale).astype(int)
            em_img_resized = cv2.resize(rotated, (side, side))
            mask = em_img_resized[..., 3:] / 255.0
            x = int(x + eyes[0][0] + eyes[0][2] / 2 - new_eye[0])
            y = int(y + eyes[0][1] + eyes[0][3] / 3 - new_eye[1])
            # pixels to trim if the image goes over one of the frame edges
            cut = [max(0, 0-x), max(0, 0-y), max(0, (x+side)-frame.shape[1]), max(0, (y+side)-frame.shape[0])]
            x += cut[0];  y += cut[1];  w = side - cut[0] - cut[2];  h = side - cut[1] - cut[3]
            # Apply the alpha mask so only the circular emoji appears over the frame image
            frame[y:y+h, x:x+w, :] = (1.0 - mask[cut[1]:cut[1]+h, cut[0]:cut[0]+w]) * frame[y:y+h, x:x+w] + mask[cut[1]:cut[1]+h, cut[0]:cut[0]+w] * em_img_resized[cut[1]:cut[1]+h, cut[0]:cut[0]+w, :3]

    cv2.imshow("VT-moji", frame)
    if cv2.waitKey(1) == 13:  # Enter key kills program
        break

cap.release()
cv2.destroyAllWindows()
