from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.preprocessing import image
import numpy as np


def define_inception_feature_model():
    base_model = InceptionV3(
        weights='imagenet',
        include_top=True
    )
    # We'll extract features at the final pool layer.
    model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('avg_pool').output
    )
    return model


def define_lstm(sequence_length, feature_length, num_classes):
    model = Sequential()
    model.add(LSTM(feature_length, return_sequences=False,
                   input_shape=(sequence_length, feature_length),
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def extract_image_feature(feature_model, image_path):
    img = image.load_img(image_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Get the prediction.
    features = feature_model.predict(x)
    features = features[0]
    return features
