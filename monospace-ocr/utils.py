from pathlib import Path

import cv2
import numpy

def get_color(arr):
    return []

def features(cell):
    bw = to_black_and_white(cell)
    nz = numpy.nonzero(bw)
    row_l, row_r = numpy.min(nz[0]), numpy.max(nz[0])
    col_l, col_r = numpy.min(nz[1]), numpy.max(nz[1])
    return (row_r - row_l + 1) / (col_r - col_l + 1)

def file_cnt(path):
    path = Path(path)
    return sum(1 for entry in path.iterdir() if entry.is_file())

def get_bounding_box(image, background):
    text_indices = numpy.nonzero(image - background)
    if len(text_indices) == 0 or len(text_indices[0]) == 0 or len(text_indices[0]) == 0:
        return None
    row_l, row_r = numpy.min(text_indices[0]), numpy.max(text_indices[0])
    col_l, col_r = numpy.min(text_indices[1]), numpy.max(text_indices[1])
    return image[row_l:(row_r + 1), col_l:(col_r + 1)]

def to_black_and_white(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)[1]

def is_monochrome_aux(image):
    return len(numpy.nonzero(image - image[0][0])[0]) == 0

def is_monochrome(image):
    height = image.shape[0]
    width = image.shape[1]
    return is_monochrome_aux(image[(height // 5):-(height // 5), (width // 5):-(width // 5)])

def get_background_black_and_white(image):
    non_corner_color_indices = numpy.transpose(numpy.nonzero(image - image[0][0]))
    are = image.shape[0] * image.shape[1]
    if len(non_corner_color_indices) > are / 2:
        index = non_corner_color_indices[0]
        return image[index[0]][index[1]]
    else:
        return image[0][0]

def get_feature_vector(symbol):
    pixels = numpy.nonzero(symbol)
    row_l, row_r = numpy.min(pixels[0]), numpy.max(pixels[0])
    col_l, col_r = numpy.min(pixels[1]), numpy.max(pixels[1])
    height = row_r - row_l + 1
    width = col_r - col_l + 1

    return numpy.array([
        height / width, # aspect ratio
        height / symbol.shape[0], # height percentage
        width / symbol.shape[1], # width percentage
        numpy.mean(pixels[0]) / symbol.shape[0], # relative vertical center
        numpy.mean(pixels[1]) / symbol.shape[1], # relative horizontal center
    ])

def train_test_split(arr, partition_size, current_partition):
    partition_length = len(arr) // partition_size
    start = current_partition * partition_length
    end = start + partition_length
    test_split = arr[start:end]
    train_split = numpy.concat((arr[0:start], arr[end:]))
    return train_split, test_split