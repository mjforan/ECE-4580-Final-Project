import pandas as pd
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

'''
Read in the csv data from the fer2013 dataset and load it into a pandas dataframe

'''
def reshape_to_image(pixStr, target):
    # the pixels column contains string of pixel values
    arr = pixStr.split()

    # convert to numpy array
    arr = np.array(arr)

    # normalize image
    arr = arr / 255.0

    # reshape 1d original data to a 48x48 image
    # (-1) will infer number of rows from (48) columns
    return np.reshape(arr, (-1, 48)), target

EMOTIONS = {
    0:"angry",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"sad",
    5:"surprise",
    6:"neutral"
}

train_dir = "./data/train.csv"
batch_size=256

df = pd.read_csv(train_dir)

# convert pixel strings into 2d numpy arrays
images = df['pixels'].tolist()
faces = []
for image in images:
    face = reshape_to_image(image)
    faces.append(face)

emotions = pd.get_dummies(target).as_matrix()

x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=69)


