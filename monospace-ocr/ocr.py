import cv2
import numpy
import ocr_model
import os
import text_grid

def image_to_text(image_path):
    if not os.path.exists(image_path):
        raise Exception('Image', image_path, 'does not exist.')
    image = cv2.imread(image_path)
    vertical_splits, horizontal_splits, cells = text_grid.parse_cells(image)


    cells = list(enumerate(map(lambda it: ocr_model.normalize(it, is_grayscale = True, for_model = True), cells)))

    width = len(vertical_splits) - 1
    height = len(horizontal_splits) - 1

    result = numpy.array([' ' for _ in range(len(cells))])

    non_empty_cells = list(filter(lambda cell: cell[1] is not None, cells))

    for i, cell in enumerate(non_empty_cells):
        cv2.imwrite('model/0/' + str(i) + '.jpg', cell[1] * 255)

    predictions = ocr_model.predict(non_empty_cells)

    for i, answer in enumerate(predictions):
        result[non_empty_cells[i][0]] = answer

    for i in range(height):
        line = ''
        for j in range(width):
            line = line + result[i * width + j]
        print(line)
