import os.path

import cv2
import numpy
import ocr_model
import text_grid
import image_codec

def image_to_text(image):
    ver_w, hor_w, vertical_splits, horizontal_splits, cells = text_grid.parse_cells(image)

    cells_data = list(enumerate(map(lambda it: ocr_model.normalize(it, is_grayscale = False, for_model = True), cells)))

    width = len(vertical_splits)
    height = len(horizontal_splits)

    result = numpy.array([' ' for _ in range(len(cells_data))])

    non_empty_cells = list(filter(lambda cell: cell[1][3] is not None, cells_data))

    predictions = ocr_model.predict(non_empty_cells)

    for i, answer in enumerate(predictions):
        result[non_empty_cells[i][0]] = answer

    for i in range(height):
        line = ''
        for j in range(width):
            line = line + result[i * width + j]
        print(line)
