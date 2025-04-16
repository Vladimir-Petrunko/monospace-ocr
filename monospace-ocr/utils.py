from pathlib import Path

import cv2
import numpy

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
    height = symbol.shape[0]
    width = symbol.shape[1]
    area = height * width
    black = numpy.nonzero(symbol - 255)

    bounding_box_width = numpy.max(black[1]) - numpy.min(black[1]) + 1
    bounding_box_height = numpy.max(black[0]) - numpy.min(black[0]) + 1
    x, y = black[1], black[0]
    x = x - width / 2
    x = x / (width / 2)
    y = y - height / 2
    y = y / (height / 2)

    return numpy.array([
        len(black[0]) / area,
        bounding_box_width / width,
        bounding_box_height / height,
        numpy.mean(x),
        numpy.mean(y),
        numpy.mean(x * x),
        numpy.mean(y * y),
        numpy.mean(x * y)
    ])

def train_test_split(arr, partition_size, current_partition):
    partition_length = len(arr) // partition_size
    start = current_partition * partition_length
    end = start + partition_length
    test_split = arr[start:end]
    train_split = numpy.concat((arr[0:start], arr[end:]))
    return train_split, test_split

def area(box):
    return (box[1] - box[0] + 1) * (box[3] - box[2] + 1)

def intersection_area(box_a, box_b):
    row_l = max(box_a[0], box_b[0])
    row_r = min(box_a[1], box_b[1])
    col_l = max(box_a[2], box_b[2])
    col_r = min(box_a[3], box_b[3])
    if row_l > row_r or col_l > col_r:
        return 0
    return area((row_l, row_r, col_l, col_r))

def union_area(box_a, box_b):
    return area(box_a) + area(box_b) - intersection_area(box_a, box_b)

def intersection_over_union(box_a, box_b):
    return intersection_area(box_a, box_b) / union_area(box_a, box_b)