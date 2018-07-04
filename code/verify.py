from keras.models import load_model
import glob
import numpy as np
import os
import time
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


MODEL_PATH = '../checkpoints/lstm.010-1.141.hdf5'


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


model = load_model(MODEL_PATH)


y_val_true = np.argmax(y_val, axis=1)
y_val_pred = model.predict_classes(X_val)


matrix = confusion_matrix(y_val_true, y_val_pred)
print(matrix)
print(class_names)


subclass_to_class = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
map_to_major_class = lambda subclass: subclass_to_class[subclass]
vfunc = np.vectorize(map_to_major_class)
y_major_val_true = vfunc(y_val_true)
y_major_val_pred = vfunc(y_val_pred)
major_class_matrix = confusion_matrix(y_major_val_true, y_major_val_pred)
print(major_class_matrix)
print(['MULTI_TOUCH', 'NO_TOUCH', 'ONE_TOUCH'])


from sklearn.metrics import accuracy_score
subclass_accuracy = accuracy_score(y_val_true, y_val_pred)
class_accuracy = accuracy_score(y_major_val_true, y_major_val_pred)
print(subclass_accuracy)
print(class_accuracy)
