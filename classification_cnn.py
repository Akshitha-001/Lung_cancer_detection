from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

def cnn_classifier(input_shape=(128,128,3)):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
