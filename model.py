import tensorflow as tf
from tensorflow.keras import layers, models

def build_load_model():
    try:
        model = tf.keras.models.load.model('digit_prediction_mnsit_model.h5')
        print('Loaded existing model...')
    except:
        print('No saved model found, creating a new one')

        # Build model and add layers
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    return model