import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import math
from keras.utils import plot_model
flags = tf.app.flags
FLAGS = flags.FLAGS 

flags.DEFINE_integer('epochs', 20, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")
flags.DEFINE_float('steering_adjustment', 0.27, "Adjustment angle.")


colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('driving_log.csv', skiprows=[0], names=colnames)
center = data.center.tolist()
center_recover = data.center.tolist() 
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()
steering_recover = data.steering.tolist()

center, steering = shuffle(center, steering)
center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 100) 

#Data pre-processing and correction
drive_straight, drive_left, drive_right = [], [], []
a_straight, a_left, a_right = [], [], []
for i in steering:
  index = steering.index(i)
  if i > 0.15:
    drive_right.append(center[index])
    a_right.append(i)
  if i < -0.15:
    drive_left.append(center[index])
    a_left.append(i)
  else:
    drive_straight.append(center[index])
    a_straight.append(i)

ds_size, dl_size, dr_size = len(drive_straight), len(drive_left), len(drive_right)
main_size = math.ceil(len(center_recover))
l_xtra = ds_size - dl_size
r_xtra = ds_size - dr_size
# Random list of indices for left and right recovery images
indice_L = random.sample(range(main_size), l_xtra)
indice_R = random.sample(range(main_size), r_xtra)

#Recovery
for i in indice_L:
  if steering_recover[i] < -0.15:
    drive_left.append(right[i])
    a_left.append(steering_recover[i] - FLAGS.steering_adjustment)

for i in indice_R:
  if steering_recover[i] > 0.15:
    drive_right.append(left[i])
    a_right.append(steering_recover[i] + FLAGS.steering_adjustment)

## COMBINE TRAINING IMAGE NAMES AND ANGLES INTO X_train and y_train ##  
X_train = drive_straight + drive_left + drive_right
y_train = np.float32(a_straight + a_left + a_right)

### PART 2: ARGUMENTATION AND PREPROCESSING ###

# Generate random brightness function, produce darker transformation 
def random_brightness(image, count):
    #Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if count == 1:
      cv2.imwrite('random_image.jpg', new_img)
    return new_img 

# Flip image around vertical axis
def flip(image, angle, count):
  new_image = cv2.flip(image,1)
  if count == 1:
      cv2.imwrite('flipped_image.jpg', new_image)
  new_angle = angle*(-1)
  return new_image, new_angle

# Crop image to remove the sky and driving deck, resize to 64x64 dimension 
def crop_resize(image):
  cropped = cv2.resize(image[60:140,:], (64,64))
  return cropped

## GENERATORS FOR KERAS fit_generator() ##

# Training generator: shuffle training data before choosing data, pick random training data to feed into batch at each "for" loop.
# Apply random brightness, resize, crop into the chosen sample. Add some small random noise for chosen angle.

def generator_data(batch_size):
    flip_count = 0
    b_count = 0
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
        data, angle = shuffle(X_train, y_train)
        for i in range(batch_size):
          choice = int(np.random.choice(len(data),1))
          batch_train[i] = crop_resize(random_brightness(mpimg.imread(data[choice].strip()),
          b_count))
          b_count += 1
          batch_angle[i] = angle[choice]*(1+ np.random.uniform(-0.10,0.10))
          #Flip random images#
          flip_coin = random.randint(0,1)
          if flip_coin == 1:
            batch_train[i], batch_angle[i] = flip(batch_train[i],
            batch_angle[i], flip_count)
            flip_count += 1

        yield batch_train, batch_angle

# Validation generator: pick random samples. Apply resizing and cropping on chosen samples        
def generator_valid(data, angle, batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
      data, angle = shuffle(data,angle)
      for i in range(batch_size):
        rand = int(np.random.choice(len(data),1))
        batch_train[i] = crop_resize(mpimg.imread(data[rand].strip()))
        batch_angle[i] = angle[rand]
      yield batch_train, batch_angle

### PART 3: TRAINING ###
def main(_):
  data_generator = generator_data(FLAGS.batch_size)
  valid_generator = generator_valid(X_valid, y_valid, FLAGS.batch_size)

# Training Architecture: inspired by NVIDIA architecture #
  input_shape = (64,64,3)
  model = Sequential()
  model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
  model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(80, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(40, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(16, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(10, W_regularizer = l2(0.001)))
  model.add(Dense(1, W_regularizer = l2(0.001)))
  adam = Adam(lr = 0.0001)
  model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
  model.summary()
  plot_model(model, to_file='model.png')
  model.fit_generator(data_generator, samples_per_epoch = math.ceil(len(X_train)), nb_epoch=FLAGS.epochs, validation_data = valid_generator, nb_val_samples = len(X_valid))

  print('Done Training')
  model.save('model.h5')

if __name__ == '__main__':
  tf.app.run()
