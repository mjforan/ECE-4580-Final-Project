import pandas as pd
import numpy as np
from tensorflow.keras.utils import Sequence
import cv2

class FaceDataset():

    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def load_data(self):
        faces = []
        pixels = self.df['pixels'].tolist()
        for sequence in pixels:
            face = [int(pixel) for pixel in sequence.split()]
            face = np.asarray(face).reshape(48, 48)
            face = cv2.resize(face.astype('uint8'), (48, 48))
            faces.append(face.astype('float32'))

        #expand the channel dimension of each image
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        #convert labels to categorical matrix
        emotions = pd.get_dummies(self.df['emotion']).values

        return faces, emotions