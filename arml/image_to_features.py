import glob, os, shutil, time
import numpy as np
from tqdm import tqdm
from config import *
from models import define_inception_feature_model, extract_image_feature
from util import rescale_list


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


feature_model = define_inception_feature_model()
for image_file_prefix_path in tqdm(image_file_prefix_path_to_image_file_paths):
    image_file_paths = image_file_prefix_path_to_image_file_paths[image_file_prefix_path]
    downsampled_image_file_paths = rescale_list(image_file_paths, SEQUENCE_LENGTH)
    video_sequence_features = []
    for image_file_path in downsampled_image_file_paths:
        features = extract_image_feature(feature_model, image_file_path)
        video_sequence_features.append(features)
    video_identifier = image_file_prefix_path[image_file_prefix_path.rindex('/')+1:]
    feature_class_path = image_file_prefix_path_to_feature_class_path[image_file_prefix_path]
    video_feature_file_path = os.path.join(feature_class_path, video_identifier + '-features')
    # Save the sequence.
    np.save(video_feature_file_path, video_sequence_features)
print('Took {0} seconds to extract features from images.'.format(time.time()-start))
# Took 202226.23611974716 seconds to extract features from images.
