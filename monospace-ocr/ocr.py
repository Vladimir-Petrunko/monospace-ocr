import cv2
import os
import text_grid

def image_to_text(image_path):
    if not os.path.exists(image_path):
        raise Exception('Image', image_path, 'does not exist.')
    image = cv2.imread(image_path)
    cells = text_grid.parse_cells(image)
