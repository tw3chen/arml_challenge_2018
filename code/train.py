import glob
import numpy as np
import os
import time
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from tqdm import tqdm


FEATURE_FOLDER_PATH = '../feature_data/train'


def get_class_names():
    class_names = []
    for class_folder_path in glob.glob(os.path.join(FEATURE_FOLDER_PATH, '*')):
        class_name = class_folder_path[class_folder_path.rindex('/')+1:]
        class_names.append(class_name)
        class_names = sorted(class_names)
    return class_names


start = time.time()
class_names = get_class_names()
X = []
y = []
for class_folder_path in tqdm(glob.glob(os.path.join(FEATURE_FOLDER_PATH, '*'))):
    class_name = class_folder_path[class_folder_path.rindex('/') + 1:]
    class_encoded = class_names.index(class_name)
    class_one_hot = to_categorical(class_encoded, len(class_names))
    for feature_file_path in glob.glob(os.path.join(class_folder_path, '*')):
        video_feature_sequence = np.load(feature_file_path)
        X.append(video_feature_sequence)
        y.append(class_one_hot)
X = np.array(X)
y = np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
print('Took {0} seconds to load all video features into memory.'.format(time.time()-start))


def define_lstm(sequence_length, feature_length, num_classes):
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False,
                   input_shape=(sequence_length, feature_length),
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


model = define_lstm(20, 2048, len(class_names))
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])

model_name = 'lstm'
tb = TensorBoard(log_dir=os.path.join('../log_data', model_name))
early_stopper = EarlyStopping(patience=2)
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('../log_data', model_name + '-' + 'training-' + \
                                    str(timestamp) + '.log'))
checkpointer = ModelCheckpoint(
    filepath=os.path.join('../checkpoints', model_name + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
    verbose=1, save_best_only=True)


start = time.time()
model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val),
          verbose=1, callbacks=[tb, early_stopper, csv_logger, checkpointer], epochs=1000)
print('Took {0} seconds to train model.'.format(time.time()-start))


y_val_true = np.argmax(y_val, axis=1)
y_val_pred = model.predict_classes(X_val)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_val_true, y_val_pred)
print(matrix)
print(class_names)
