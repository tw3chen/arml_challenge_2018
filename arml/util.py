import glob, os
from config import *


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


def get_class_names():
    class_names = []
    for class_folder_path in glob.glob(os.path.join(FEATURE_FOLDER_PATH, '*')):
        class_name = class_folder_path[class_folder_path.rindex('/')+1:]
        class_names.append(class_name)
        class_names = sorted(class_names)
    return class_names
