import cv2
import numpy
import ocr_model
import os
import text_grid
import image_codec
from PIL import Image

def image_to_text(image):
    ver_w, hor_w, vertical_splits, horizontal_splits, cells = text_grid.parse_cells(image)

    cells_data = list(enumerate(map(lambda it: ocr_model.normalize(it, is_grayscale = False, for_model = True), cells)))

    width = len(vertical_splits)
    height = len(horizontal_splits)

    result = numpy.array([' ' for _ in range(len(cells_data))])
    foregrounds = []
    backgrounds = []

    for ce in cells_data:
        backgrounds.append(ce[1][1])
        foregrounds.append(ce[1][2])

    non_empty_cells = list(filter(lambda cell: cell[1][3] is not None, cells_data))

    for i, cell in enumerate(non_empty_cells):
        cv2.imwrite('model/0/' + str(i) + '.jpg', cell[1][3] * 255)

    predictions = ocr_model.predict(non_empty_cells)

    for i, answer in enumerate(predictions):
        result[non_empty_cells[i][0]] = answer

    for i in range(height):
        line = ''
        for j in range(width):
            line = line + result[i * width + j]
        print(line)

    region = image_codec.Region(
        coordinates = (0, 0),
        split_size = (ver_w, hor_w),
        splits = (vertical_splits, horizontal_splits),
        symbols = result,
        foregrounds = foregrounds,
        backgrounds = backgrounds
    )

    with open('ans.txt', 'wb') as ff:
        ff.write(image_codec.encode([region]))

    img = Image.new(mode = 'RGB', size = (800, 800))
    img = image_codec.generate_region(region, img)
    img.save('res.png')
