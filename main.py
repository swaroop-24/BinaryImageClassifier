import os
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import tensorflow as tf
import cv2
import numpy as np


data = tf.keras.utils.image_dataset_from_directory("Coral")
data = data.map(lambda x, y: (x/255, y))
data.as_numpy_iterator().next()
train = data.take(int(len(data)*.7))
validation = data.skip(int(len(data) * .7)).take(int(len(data) * .3))

model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
history = model.fit(train, epochs=25, validation_data=validation, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='logs')])
model.save(os.path.join('models', 'imageclassifier.h5'))


new_model = load_model('models/imageclassifier.h5')
image = cv2.imread('test2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sol = new_model.predict(np.expand_dims(tf.image.resize(image, (256, 256)) / 255, 0))
if sol > 0.5:
    print('Predicted class is Positive')
else:
    print('Predicted class is Negative')

