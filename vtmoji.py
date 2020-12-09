import pandas as pd
import tensorflow as tf
import numpy as np
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from model import create_model
from dataloader import FaceDataset

'''
Read in the csv data from the fer2013 dataset and load it into a pandas dataframe

'''
def string_to_image(pix_str):
    # the pixels column contains string of pixel values
    face = [int(pixel) for pixel in pix_str.split()]
    # convert to numpy array
    face = np.asarray(face).reshape(48, 48)
    #arr = np.array(arr).astype('float32')

    # normalize image
    face = face / 255.0

    # reshape 1d original data to a 48x48 image
    # (-1) will infer number of rows from (48) columns
    return face

EMOTIONS = {
    0:"angry",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"sad",
    5:"surprise",
    6:"neutral"
}

data_fpath = "./data/fer2013/fer2013.csv"

# get dataset
dataset = FaceDataset(data_fpath)
faces, emotions = dataset.load_data()


data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=.1,
        horizontal_flip=True
)



# get train-test split
x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)

# split training once more into validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=69)

num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48

model = create_model(width=width, height=height, num_features=num_features, num_labels=num_labels)

model.summary()

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

# call custom function to get a list of callback objects
callbacks = []
# add callbacks
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=True)
callbacks.append(lr_reducer)

# save logs in separate directories
tb_logdir= os.path.join(os.getcwd(), 'training_logs')
i = 0 
while os.path.exists(os.path.join(tb_logdir, str(i))):
    i += 1
tb_logdir = os.path.join(tb_logdir, str(i))
os.makedirs(tb_logdir)

tb = TensorBoard(log_dir=tb_logdir)
callbacks.append(tb)

#checkpointer = ModelCheckpoint(tb_logdir, monitor='val_loss', verbose=True, save_best_only=True)
#callbacks.append(checkpointer)

flower = data_generator.flow(x_train, y_train, batch_size, shuffle=True)

model.fit_generator(flower, steps_per_epoch=len(x_train) / batch_size, epochs=epochs, verbose=True, callbacks=callbacks, validation_data=(x_val, y_val))

# train the model
'''
model.fit_generator(data_generator.flow((np.array(x_train), np.array(y_train), 
    batch_size=batch_size,
    epochs=epochs,
    verbose=True,
    validation_data=(x_test, y_test), 
    shuffle=True,
    callbacks=callbacks
)))
'''

scores = model.evaluate(np.array(x_test), np.array(y_test), batch_size=batch_size)
print(f"Loss: {scores[0]}")
print(f"Accuracy: {scores[1]}")

model.save(os.path.join(tb_logdir, 'trained'))