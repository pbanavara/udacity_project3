import csv
import cv2
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D

def preprocess_input():
    lines = []
    with open("../data_assignment3/attempt12/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    for line in lines:
        center_source_path = line[0]
        left_source_path = line[1]
        right_source_path = line[2]
        center_filename = center_source_path.split('/')[-1]
        left_filename = left_source_path.split('/')[-1]
        right_filename = right_source_path.split('/')[-1]
        center_current_path = "../data_assignment3/attempt12/IMG/" + center_filename
        left_current_path = "../data_assignment3/attempt12/IMG/" + left_filename
        right_current_path = "../data_assignment3/attempt12/IMG/" + right_filename
        center_image = cv2.imread(center_current_path)
        left_image = cv2.imread(left_current_path)
        right_image = cv2.imread(right_current_path)
        images.append(center_image)
        images.append(left_image)
        images.append(right_image)
        steering_center = float(line[3])
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        measurements.append(steering_center)
        measurements.append(steering_left)
        measurements.append(steering_right)
    
    return images, measurements

images, measurements = preprocess_input()
X_train = np.array(images)
y_train = np.array(measurements)
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = "relu" ))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = "relu" ))
model.add(Dropout(0.5))
model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation = "relu" ))
model.add(Convolution2D(64, 3, 3, activation = "relu" ))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation = "relu" ))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 25)
model.save('../behavior_model_attempt12.h5')
