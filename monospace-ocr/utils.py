from pathlib import Path

import cv2
import numpy
from skimage.morphology import flood
from dto import CellDto

# Sorry for the naming :)
# This variable is used during normalization. It indicates whether the rightmost column of the last symbol
# was included in the symbol or not. If so, the leftmost column is cut from the next symbol. This is needed
# because two adjacent symbols share a common column.
lefteroo = False

def file_cnt(path):
    """
    Returns the number of files in the given path.
    :param path: the path
    :return: the number of files in the path
    """
    path = Path(path)
    return sum(1 for entry in path.iterdir() if entry.is_file())

def to_black_and_white(image):
    """
    Performs Otsu binarization on the input image.
    Precondition: image is in grayscale format.
    :param image: the input image
    :return: the binarized version of the input image
    """
    return cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)[1]

def get_background_black_and_white(image, is_binarized):
    """
    Gets background color of symbol as maximum frequency color (black or white).
    Precondition: image is in grayscale format.
    :param image: the input image
    :param is_binarized: True if the image is binarized, False otherwise
    :return: 0 or 255, depending on the determined background color
    """
    orig = image.copy()
    if not is_binarized:
        orig = to_black_and_white(orig)
    non_corner_color_indices = numpy.transpose(numpy.nonzero(orig - orig[0][0]))
    if len(non_corner_color_indices) > (image.shape[0] * image.shape[1]) / 2:
        index = non_corner_color_indices[0]
        return orig[index[0]][index[1]]
    else:
        return orig[0][0]


def get_feature_vector(symbol):
    """
    Returns a feature vector from the given symbol.
    Idea taken from https://link.springer.com/article/10.1007/bf00114162
    Preconditions:
    1. Symbol is normalized, white on black background
    2. Symbol width ~ true character width
    3. Symbol height ~ true character height times 2
    Features for retrieval:
    1. Width of bounding box, divided by symbol width
    2. Height of bounding box, divided by symbol height
    3. Width of bounding box, divided by height of bounding box
    4. True symbol area (i.e. white pixel count), divided by area of bounding box
    5. Relative vertical center, normalized to range [-1, 1]
    6. Relative horizontal center, normalized to range [-1, 1]
    :param symbol: input symbol
    :return: feature vector, as described above
    """

    pixels = numpy.nonzero(symbol)

    # Get coordinates of bounding box
    row_l, row_r = numpy.min(pixels[0]), numpy.max(pixels[0])
    col_l, col_r = numpy.min(pixels[1]), numpy.max(pixels[1])

    height = row_r - row_l + 1
    width = col_r - col_l + 1

    feature_vector = [
        width / symbol.shape[1],
        height / symbol.shape[0],
        width / height,
        len(pixels[0]) / (width * height),
        numpy.mean(pixels[0]) / symbol.shape[0],
        numpy.mean(pixels[1]) / symbol.shape[1]
    ]

    return numpy.array(feature_vector)

def normalize(cell, is_grayscale, for_model, target_size = 12):
    """
    Performs cell normalization. This process consists of the following steps:
    1. Constrast increase and Otsu binarization
    2. Noise removal via DFS on edges
    3. Determination of background and symbol color + feature vector
    4. Rescaling to square size for ML model input
    THIS FUNCTION OPERATES IN-PLACE.
    :param cell: the input cell
    :param is_grayscale: whether the cell is initially grayscale
    :param for_model: whether the resulting cell should be divided by 255 for ML model input
    :param target_size: the target width and height of the size
    :return: a tuple of 4 elements: (feature vector, background color, symbol color, normalized cell)
    """

    cell = cell.astype('uint8')
    orig = cell.copy()
    if not is_grayscale:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Delete leftmost column if it was used in last symbol
    global lefteroo
    if lefteroo:
        cell = cell[:, 1:]
        orig = orig[:, 1:]
    lefteroo = False

    light = False
    # Symbol must be white on black background
    if get_background_black_and_white(cell, is_binarized = False) == 255:
        light = True
        cell = 255 - cell

    # Technical resize
    coefficient = 2
    orig_inc = cv2.resize(orig, (orig.shape[1] * coefficient, orig.shape[0] * coefficient), interpolation = cv2.INTER_LANCZOS4)
    cell = cv2.resize(cell, (cell.shape[1] * coefficient, cell.shape[0] * coefficient), interpolation = cv2.INTER_LANCZOS4)

    min_color, max_color = numpy.min(cell), numpy.max(cell)
    if max_color == min_color:
        # Nothing on symbol
        background = numpy.average(orig, axis = (0, 1))
        return CellDto(None, background, background, None)
    cell = cell - min_color
    cell = cell * (255 / (max_color - min_color))

    cell = cell.astype('uint8')
    binarized = to_black_and_white(cell)
    pixels = numpy.nonzero(binarized)

    background = [0, 0, 0]
    cnt = 0
    for i in range(0, orig_inc.shape[0], coefficient * 2):
        for j in range(0, orig_inc.shape[1], coefficient * 2):
            if binarized[i][j] == 0:
                background = background + orig_inc[i][j]
                cnt = cnt + 1
    background = background / cnt

    # Clear almost-filled white rows
    white_remaining = len(pixels[0])
    for i in range(cell.shape[0] // 4):
        for coeff in [-1, 1]:
            if white_remaining > numpy.sum(binarized[i * coeff, :]) >= 255 * (cell.shape[1] * 0.8):
                cell[i * coeff, :] = 0
                binarized[i * coeff, :] = 0
                white_remaining = white_remaining - numpy.sum(cell[i * coeff, :]) // 255

    # Clear unconnected parts of edge columns
    mask = numpy.zeros(cell.shape)
    for i in range(0, cell.shape[0], 2):
        for j in range(2 * coefficient, cell.shape[1] - 2 * coefficient):
            if binarized[i][j] == 255 and mask[i][j] == 0:
                res = flood(binarized, (i, j))
                mask = numpy.logical_or(mask, res)
    mask[:, (2 * coefficient):(cell.shape[1] - 2 * coefficient)] = 1

    foreground = [0, 0, 0] if not light else [255, 255, 255]
    for i in range(0, orig_inc.shape[0], coefficient * 2):
        for j in range(0, orig_inc.shape[1], coefficient * 2):
            if binarized[i][j] != 0 and not light:
                foreground[0] = max(foreground[0], orig_inc[i][j][0])
                foreground[1] = max(foreground[1], orig_inc[i][j][1])
                foreground[2] = max(foreground[2], orig_inc[i][j][2])
            if binarized[i][j] != 0 and light:
                foreground[0] = min(foreground[0], orig_inc[i][j][0])
                foreground[1] = min(foreground[1], orig_inc[i][j][1])
                foreground[2] = min(foreground[2], orig_inc[i][j][2])

    cell = cell * mask
    binarized = binarized * mask

    if len(numpy.nonzero(binarized[:, -coefficient:])[0]) > 0:
        lefteroo = True

    pixels = numpy.nonzero(binarized)

    if len(pixels[0]) == 0:
        return CellDto(None, background, background, None)

    feature_vector = get_feature_vector(binarized)

    # Resize symbol to required dimensions
    row_l, row_r = numpy.min(pixels[0]), numpy.max(pixels[0])
    col_l, col_r = numpy.min(pixels[1]), numpy.max(pixels[1])
    cell = cell[row_l:(row_r + 1), col_l:(col_r + 1)]
    extra_width = cell.shape[1] - cell.shape[0]
    cell = numpy.pad(cell, [(0, max(extra_width, 0)), (0, max(-extra_width, 0))])
    cell = cv2.resize(cell, (target_size, target_size), interpolation = cv2.INTER_LANCZOS4)

    cell = cell.astype('uint8')
    cell = to_black_and_white(cell)

    return CellDto(feature_vector, background, foreground, cell / 255 if for_model else cell)