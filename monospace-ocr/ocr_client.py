import cv2
from rich.region import Region

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
    vertical_size, horizontal_size, vertical_splits, horizontal_splits, cells = text_grid.parse_cells(image)
    cells_data = list(enumerate(map(
        lambda cell: utils.normalize(cell, is_grayscale = False, for_model = True), cells
    )))

    width = len(vertical_splits) - 1
    height = len(horizontal_splits) - 1

    result = numpy.array([' ' for _ in range(len(cells_data))])

    non_empty_cells = list(filter(lambda it: it[1].cell is not None, cells_data))
    predictions = ocr_model.predict(non_empty_cells)

    for i, answer in enumerate(predictions):
        result[non_empty_cells[i][0]] = answer

    result = ocr_model.post_analysis(result, height, width)
    result = numpy.reshape(result, (height, width))
    return (vertical_size, horizontal_size), (vertical_splits, horizontal_splits), cells_data, result

def image_to_text(image):
    _, _, _, result = parse(image)
    return '\n'.join(''.join(str(x) for x in row) for row in result)

def encode(image, font = 'Consolas', output_file = 'output.eva'):
    (vertical_size, horizontal_size), (vertical_splits, horizontal_splits), cells_data, result = parse(image)
    coordinates = (vertical_splits[0], horizontal_splits[0])
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
    encoded = image_codec.encode(image, [region])
    with open(output_file, 'wb') as file:
        file.write(encoded)

def decode(input_file = 'output.eva', output_file = 'image.png'):
    with open(input_file, 'rb') as file:
        encoded = file.read()
    image_codec.decode(encoded, output_file)
