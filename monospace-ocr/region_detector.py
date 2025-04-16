import cv2
import numpy
import utils
from skimage.morphology import flood

BLOCK_SIZE = 5
SIMILARITY_DELTA = 100
FLOOD_FILL_DELTA = 5
MINIMUM_AREA_PERCENT = 0.1
MINIMUM_SPARSITY_PERCENT = 0.3

def similar(image, row_l, row_r, col_l, col_r):
    sub_image = image[row_l:row_r, col_l:col_r]
    return len(sub_image[abs(sub_image - sub_image[0][0]) > SIMILARITY_DELTA]) == 0

def intersects(a, b):
    return utils.intersection_area(a, b) > 0

def detect_regions(image):
    sharpening_kernel = numpy.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.filter2D(image, -1, sharpening_kernel)

    width, height = image.shape[1], image.shape[0]
    visited = numpy.zeros((height // BLOCK_SIZE + 1, width // BLOCK_SIZE + 1))

    boxes = []

    for i in range(0, height, BLOCK_SIZE):
        for j in range(0, width, BLOCK_SIZE):
            if visited[i // BLOCK_SIZE][j // BLOCK_SIZE] == 1:
                continue
            if not similar(image, i, i + BLOCK_SIZE, j, j + BLOCK_SIZE):
                continue
            mask = flood(image, (i, j), tolerance = FLOOD_FILL_DELTA)
            nonzero = numpy.nonzero(mask)
            y = nonzero[0] // BLOCK_SIZE
            x = nonzero[1] // BLOCK_SIZE
            visited[y, x] = 1
            area = len(nonzero[0]) / (width * height)
            row_l, row_r = numpy.min(nonzero[0]), numpy.max(nonzero[0])
            col_l, col_r = numpy.min(nonzero[1]), numpy.max(nonzero[1])
            sparsity = len(nonzero[0]) / ((row_r - row_l + 1) * (col_r - col_l + 1))
            if area > MINIMUM_AREA_PERCENT and sparsity > MINIMUM_SPARSITY_PERCENT:
                has_seen = False
                for box in boxes:
                    if intersects((row_l, row_r, col_l, col_r), box):
                        has_seen = True
                        break
                if not has_seen:
                    boxes.append((row_l, row_r, col_l, col_r))
                    image[mask == 1] = 50
    cv2.imwrite('ll.jpg', image)
    return boxes
