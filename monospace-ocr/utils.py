from pathlib import Path

import cv2
import numpy

def file_cnt(path):
    path = Path(path)
    return sum(1 for entry in path.iterdir() if entry.is_file())

def get_bounding_box(image, background):
    background_arr = numpy.zeros(image.shape, dtype = 'int') + background
    text_indices = numpy.nonzero(image - background_arr)
    row_l, row_r = numpy.min(text_indices[0]), numpy.max(text_indices[0])
    col_l, col_r = numpy.min(text_indices[1]), numpy.max(text_indices[1])
    return image[row_l:(row_r + 1), col_l:(col_r + 1)]

def to_black_and_white(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)[1]

def get_background_black_and_white(image):
    corner_color_arr = numpy.zeros(image.shape, dtype = 'int') + image[0][0]
    non_corner_color_indices = numpy.transpose(numpy.nonzero(image - corner_color_arr))
    area = image.shape[0] * image.shape[1]
    if len(non_corner_color_indices) > area / 2:
        index = non_corner_color_indices[0]
        return image[index[0]][index[1]]
    else:
        return image[0][0]

def get_feature_vector(symbol):
    area = symbol.shape[0] * symbol.shape[1]
    black = numpy.nonzero(symbol - 255)

    black_area_percent = len(black[0]) / area
    bounding_box_width_percent = (numpy.max(black[1]) - numpy.min(black[1]) + 1) / symbol.shape[1]
    bounding_box_height_percent = (numpy.max(black[0]) - numpy.min(black[0]) + 1) / symbol.shape[0]

    return numpy.array([
        black_area_percent,
        bounding_box_width_percent,
        bounding_box_height_percent
    ])

def train_test_split(arr, partition_size, current_partition):
    partition_length = len(arr) // partition_size
    start = current_partition * partition_length
    end = start + partition_length
    test_split = arr[start:end]
    train_split = numpy.concat((arr[0:start], arr[end:]))
    return train_split, test_split
