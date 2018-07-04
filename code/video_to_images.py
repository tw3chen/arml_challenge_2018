import glob
import os
import shutil
import subprocess
import time
from tqdm import tqdm

VIDEO_FOLDER_PATH = 'video_data/train/r1'
IMAGE_FOLDER_PATH = 'image_data/train'


shutil.rmtree(IMAGE_FOLDER_PATH)
os.makedirs(IMAGE_FOLDER_PATH)


start = time.time()
video_image_path_tuples_list = []
for class_folder in glob.glob(os.path.join(VIDEO_FOLDER_PATH, '*')):
    class_name = class_folder[class_folder.rindex('/')+1:]
    image_class_path = os.path.join(IMAGE_FOLDER_PATH, class_name)
    os.makedirs(image_class_path)
    for video_file_path in glob.glob(os.path.join(class_folder, '*')):
        video_identifier = video_file_path[video_file_path.rindex('/')+1:][:-4]
        image_file_path = os.path.join(image_class_path, video_identifier + '-%04d.jpg')
        video_image_path_tuples_list.append((video_file_path, image_file_path))
print('Took {0} seconds to create class folders for images and go through all videos.'.format(time.time()-start))

start = time.time()
for video_file_path, image_file_path in tqdm(video_image_path_tuples_list):
    subprocess.call(["ffmpeg", "-i", video_file_path, image_file_path, "-loglevel", "quiet"], stdout=subprocess.DEVNULL)
print('Took {0} seconds to convert the videos into images.'.format(time.time()-start))


# took 1 hour to run