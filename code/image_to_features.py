import glob
import os
import shutil
import subprocess
import time
from tqdm import tqdm

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

IMAGE_FOLDER_PATH = 'image_data/train'
FEATURE_FOLDER_PATH = 'feature_data/train'
SEQUENCE_LENGTH = 20

shutil.rmtree(FEATURE_FOLDER_PATH)
os.makedirs(FEATURE_FOLDER_PATH)


start = time.time()
image_file_prefix_path_to_image_file_paths = {}
image_file_prefix_path_to_feature_class_path = {}
for class_folder in glob.glob(os.path.join(IMAGE_FOLDER_PATH, '*')):
    class_name = class_folder[class_folder.rindex('/')+1:]
    feature_class_path = os.path.join(FEATURE_FOLDER_PATH, class_name)
    os.makedirs(feature_class_path)
    for image_file_path in glob.glob(os.path.join(class_folder, '*')):
        image_file_prefix_path = image_file_path[:-9]
        image_file_prefix_path_to_feature_class_path[image_file_prefix_path] = feature_class_path
        if image_file_prefix_path not in image_file_prefix_path_to_image_file_paths:
            image_file_prefix_path_to_image_file_paths[image_file_prefix_path] = []
        image_file_prefix_path_to_image_file_paths[image_file_prefix_path].append(image_file_path)
    for image_file_prefix_path in image_file_prefix_path_to_image_file_paths:
        image_file_prefix_path_to_image_file_paths[image_file_prefix_path] = \
            sorted(image_file_prefix_path_to_image_file_paths[image_file_prefix_path])
print('Took {0} seconds to create class folders for features and go through all images.'.format(time.time()-start))


def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the origina list."""
    assert len(input_list) >= size
    # Get the number to skip between iterations.
    skip = len(input_list) // size
    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    # Cut off the last one if needed.
    return output[:size]


def setup_feature_model():
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


model = setup_feature_model()


def extract(image_path):
    #img = image.load_img(image_path, target_size=(299, 299)) # try without target size later
    img = image.load_img(image_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Get the prediction.
    features = model.predict(x)
    features = features[0]
    return features

for image_file_prefix_path in tqdm(image_file_prefix_path_to_image_file_paths):
    image_file_paths = image_file_prefix_path_to_image_file_paths[image_file_prefix_path]
    downsampled_image_file_paths = rescale_list(image_file_paths, SEQUENCE_LENGTH)
    video_sequence_features = []
    for image_file_path in downsampled_image_file_paths:
        features = extract(image_file_path)
        video_sequence_features.append(features)
    video_identifier = image_file_prefix_path[image_file_prefix_path.rindex('/')+1:]
    feature_class_path = image_file_prefix_path_to_feature_class_path[image_file_prefix_path]
    video_feature_file_path = os.path.join(feature_class_path, video_identifier + '-features')
    # Save the sequence.
    np.save(video_feature_file_path, video_sequence_features)
print('Took {0} seconds to extract features from images.'.format(time.time()-start))

# Took 202226.23611974716 seconds to extract features from images.