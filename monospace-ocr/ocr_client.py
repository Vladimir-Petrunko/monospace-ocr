import time

import cv2
import image_codec
import numpy
import ocr_model
import text_grid
import utils

def initialize():
    """
    Helper function for initialization.
    If not called, initialization will be performed automatically during first parsing call.
    """
    ocr_model.initialize()

def output_normalized(cells, path):
    """
    Debug function that dumps all normalized cells in the given path.
    :param cells: the cells
    :param path: the path
    """

    for i, it in enumerate(cells):
        cv2.imwrite(path + '/' + str(i) + '.png', it[1].cell * 255)

def parse(image):
    """
    Helper function that parses an image (represented as 2D NumPy array) and returns its associated data,
    as a tuple of the following elements:
    1. The vertical and horizontal grid width
    2. The vertical and horizontal split coordinates
    3. The parsed cells themselves after normalization, along with their features
    4. The recognized symbols in the cells
    :param image: the input image
    :return: a tuple, consisting of the abovementioned elements
    """
    vertical_size, horizontal_size, vertical_splits, horizontal_splits, cells = text_grid.parse_cells(image)
    cells_data = list(enumerate(map(
        lambda cell: utils.normalize(cell, is_grayscale = False, for_model = True), cells
    )))

    width = len(vertical_splits) - 1
    height = len(horizontal_splits) - 1

    result = numpy.array([' ' for _ in range(len(cells_data))])

    # We don't need to pass whitespace to OCR
    non_empty_cells = list(filter(lambda it: it[1].cell is not None, cells_data))
    predictions = ocr_model.predict(non_empty_cells)

    for i, answer in enumerate(predictions):
        result[non_empty_cells[i][0]] = answer

    # Post-analysis based on feature vector classification
    result = ocr_model.post_analysis(result, height, width)
    result = numpy.reshape(result, (height, width))
    return (vertical_size, horizontal_size), (vertical_splits, horizontal_splits), cells_data, result

def image_to_text(image):
    """
    Takes an image, and returns the parsed text on it.
    :param image: the path to the input image
    :return: the string representing the parsed image
    """
    _, _, _, result = parse(cv2.imread(image))
    return '\n'.join(''.join(str(x) for x in row) for row in result)

def encode_aux(image, parsed_data, font, output_file, keep_background):
    """
    Encodes the input image with the given parameters.
    This is an auxiliary method and is not meant to be called from outside.
    :param image: the input image as a NumPy ndarray.
    :param parsed_data: the result of calling the 'parse' method, as follows:
        (vertical_size, horizontal_size), (vertical_splits, horizontal_splits), cells_data, result
    :param font: the font with which the image is to be encoded.
    :param output_file: the path of the encoded file to be created. If it exists, contents will be overwritten.
    :param keep_background: whether the image outside the text regions is to be kept.
    """
    (vertical_size, horizontal_size), (vertical_splits, horizontal_splits), cells_data, result = parsed_data
    coordinates = (vertical_splits[0], horizontal_splits[0], vertical_splits[-1], horizontal_splits[-1])
    vertical_splits = vertical_splits - coordinates[0]
    horizontal_splits = horizontal_splits - coordinates[1]
    region = image_codec.Region(
        font = font,
        coordinates = coordinates,
        split_size = (vertical_size, horizontal_size),
        splits = (vertical_splits, horizontal_splits),
        symbols = result.flatten(),
        foregrounds = list(map(lambda cell: cell[1].foreground, cells_data)),
        backgrounds = list(map(lambda cell: cell[1].background, cells_data))
    )
    encoded = image_codec.encode(image, [region], keep_background)
    with open(output_file, 'wb') as file:
        file.write(encoded)

def encode(input_file, font, output_file, keep_background = False):
    """
    Encodes the input image with the given parameters.
    :param input_file: the path to the input image.
    :param font: the font with which the image is to be encoded.
    :param output_file: the path of the encoded file to be created. If it exists, contents will be overwritten.
    :param keep_background: whether the image outside the text regions is to be kept.
    """
    cv2_image = cv2.imread(input_file)
    encode_aux(cv2_image, parse(cv2_image), font, output_file, keep_background)

def decode(input_file, output_file):
    """
    Decodes the input image with the given parameters.
    :param input_file: the path to the encoded file.
    :param output_file: the path of the decoded image to be created. If it exists, contents will be overwritten.
    :return:
    """
    with open(input_file, 'rb') as file:
        encoded = file.read()
    image_codec.decode(encoded, output_file)
