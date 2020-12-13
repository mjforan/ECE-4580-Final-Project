# VT-Moji: Visually Trained Emoji

VT-Moji is a facial recognition and emotion detection project that uses a Haar Cascade algorithm to detect a user's face in a webcam image, Convolutional Neural Network (CNN) to determine the emotion, and overlays an appropriate emoji over the face based upon the emotion. 

To run this project, you must first install the correct packages to your local machine by calling
```
pip install -r requirements.txt
```

Then, to execute the program, call 
```
python vt-moji.py
```

## Version info

This project was executed on Window 64-bit operating system with Python 3.7 64-bit installed. Due to package dependencies, especially with Tensorflow 2, this project will not run with a Python 32-bit installation

## Training the Convolutional Neural Network

During the course of this project, we trained a convolutional neural network model which is stored in [vt-moji-0](./vt-moji-0). The below sections detail relevant information should you wish to replicate the neural network training yourself. 

### Obtaining Labelled Data
The CNN was trained using the open-source Facial Expression Recognition (FER2013) dataset available through here: [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). Kaggle is a widely used hub for machine learning competitions. If you wish to replicate the training, you must download the dataset from here, by making a free Kaggle account and verifying your email. 

### Running Training

Due to the time constraints of this project and the computational limitations of our group's computers, we took advantage of [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) to perform training.(You must be logged onto Google with an educational address to access).

Google Colaboratory, or "Colab", is essentially a platform that can execute Jupyter notebooks using Google's cloud servers, giving access to high-performance GPUs to rapidly speed up the training of neural networks. For reference, the training runs would take up to 45 minutes per epoch on a student's local computer, whereas running training on Colab had speeds of around 10 seconds per epoch. 

Thus, if you wish to replicate the training, after downloading the dataset, simply open the [training file](./training.ipynb) in Google Colab and run all the cells sequentially. 

