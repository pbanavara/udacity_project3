import csv
import cv2
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D

def preprocess_input():
    lines = []
    with open("../data_assignment3/attempt2/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        center_source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = "../data_assignment3/attempt2/IMG/" + filename
        image = cv2.imread(current_path)
        measurement = float(line[3])
        images.append(image)
        measurements.append(measurement)
    return images, measurements

images, measurements = preprocess_input()
X_train = np.array(images)
y_train = np.array(measurements)
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 10)
model.save('../behavior_model.h5')
