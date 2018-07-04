import glob
import numpy as np
import os
import time
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import *
from models import define_lstm
from util import get_class_names


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


model = define_lstm(SEQUENCE_LENGTH, 2048, len(class_names))
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])

model_name = 'lstm'
tb = TensorBoard(log_dir=os.path.join(LOG_FOLDER_PATH, model_name))
early_stopper = EarlyStopping(patience=2)
timestamp = time.time()
csv_logger = CSVLogger(os.path.join(LOG_FOLDER_PATH, model_name + '-' + 'training-' + \
                                    str(timestamp) + '.log'))
checkpointer = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_FOLDER_PATH, model_name + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
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
