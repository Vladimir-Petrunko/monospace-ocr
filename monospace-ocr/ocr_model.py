import math
import time
import cv2
import itertools
import json
import numpy
import os
import utils
from constants import CHAR_HEIGHT
from pathlib import Path
from tensorflow.keras import models
from skimage.morphology import flood

FONT_TO_LABEL = {}
LABEL_TO_FONT = {}

CHARACTER_TO_LABEL = {}
LABEL_TO_CHARACTER = {}

initialized = {'mappings': False, 'models': False}

character_models = {}

def adjust(x):
    if x < 75:
        return x
    else:
        return int(max(255, x * 1.25))

def initialize_mappings():
    global initialized
    if initialized['mappings']:
        return
    for character in range(33, 127):
        index = len(CHARACTER_TO_LABEL)
        CHARACTER_TO_LABEL[character] = index
        LABEL_TO_CHARACTER[index] = character
    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/').replace('fonts/', '')
        font_str_split = font_str.split('-')
        if font_str_split[0] not in FONT_TO_LABEL:
            index = len(FONT_TO_LABEL)
            FONT_TO_LABEL[font_str_split[0]] = index
            LABEL_TO_FONT[index] = font_str_split[0]
    initialized['mappings'] = True

def get_character_connected_components(cnt, character_counts, character_errors):
    components = []
    label_to_index = {}
    max_size = 0
    for label in range(cnt):
        for error in character_errors[label]:
            error_cnt = character_errors[label][error]
            if error_cnt / character_counts[label] < 0.002:
                continue
            index_a = label_to_index.get(label, None)
            index_b = label_to_index.get(error, None)
            if index_a is None and index_b is None:
                index = len(components)
                new_array = numpy.array([label, error])
                max_size = max(max_size, 2)
                components.append(new_array)
                label_to_index[label] = index
                label_to_index[error] = index
            elif index_a is None:
                components[index_b] = numpy.append(components[index_b], label)
                max_size = max(max_size, len(components[index_b]))
                label_to_index[label] = index_b
            elif index_b is None:
                components[index_a] = numpy.append(components[index_a], error)
                max_size = max(max_size, len(components[index_a]))
                label_to_index[error] = index_a
            elif index_a == index_b:
                continue
            else:
                for char in components[index_a]:
                    label_to_index[char] = index_b
                components[index_b] = numpy.concat((components[index_b], components[index_a]))
                components[index_a] = []
                max_size = max(max_size, len(components[index_b]))

    result = []
    current_component = []
    for component in components:
        if len(component) + len(current_component) <= max_size:
            current_component = numpy.append(current_component, component)
        else:
            result.append(current_component)
            current_component = component
    if len(current_component) > 0:
        result.append(current_component)
    return result

def normalize(cell, is_grayscale, for_model, cut = True):
    if cell.shape[0] * cell.shape[1] == 0:
        return None, None, None, None
    cell = cell.astype('uint8')

    orig = cell.copy()

    if not is_grayscale:
        bw_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        bw_cell = cell

    if utils.get_background_black_and_white(utils.to_black_and_white(bw_cell)) == 255:
        bw_cell = 255 - bw_cell

    min_color, max_color = numpy.min(bw_cell), numpy.max(bw_cell)
    bw_cell = bw_cell - min_color
    bw_cell = bw_cell * (255 / (max_color - min_color))
    bw_cell[bw_cell > 255] = 255

    bw_cell = bw_cell.astype('uint8')

    bw_cell = utils.to_black_and_white(bw_cell)

    backgrounds = []
    for i in range(cell.shape[0]):
        for j in range(cell.shape[1]):
            if bw_cell[i][j] == 0:
                backgrounds.append(orig[i][j])

    if cut:
        for i in range(cell.shape[0]):
            for j in range(cell.shape[1]):
                if cell.shape[0] / 3 <= i <= cell.shape[0] * 2 / 3 or cell.shape[1] / 3 <= i <= cell.shape[1] * 2 / 3:
                    continue
                if numpy.sum(bw_cell[i, :]) >= 255 * (cell.shape[1] - 1):
                    bw_cell[i, :] = 0
                if numpy.sum(bw_cell[:, j]) >= 255 * (cell.shape[0] - 1):
                    bw_cell[:, j] = 0

    mask = numpy.zeros(bw_cell.shape)
    for i in range(1, cell.shape[0] - 1):
        for j in range(1, cell.shape[1] - 1):
            if bw_cell[i][j] == 255 and mask[i][j] == 0:
                res = flood(bw_cell, (i, j))
                mask = numpy.logical_or(mask, res)
    mask[1:(cell.shape[0] - 1), 1:(cell.shape[1] - 1)] = 1

    for i in range(0, cell.shape[0]):
        for j in range(0, cell.shape[1]):
            if bw_cell[i][j] == 255 and mask[i][j] == 0:
                bw_cell[i][j] = 0

    foregrounds = []
    for i in range(cell.shape[0]):
        for j in range(cell.shape[1]):
            if bw_cell[i][j] == 255:
                foregrounds.append(orig[i][j])

    pixels = numpy.nonzero(bw_cell)

    if len(pixels[0]) == 0:
        return None, utils.get_color(backgrounds), utils.get_color(backgrounds), None

    features = utils.get_feature_vector(bw_cell)

    row_l, row_r = numpy.min(pixels[0]), numpy.max(pixels[0])
    col_l, col_r = numpy.min(pixels[1]), numpy.max(pixels[1])
    bw_cell = bw_cell[row_l:(row_r + 1), col_l:(col_r + 1)]
    extra_width = bw_cell.shape[1] - bw_cell.shape[0]
    if extra_width > 0:
        bw_cell = numpy.pad(bw_cell, [(0, extra_width), (0, 0)])
    else:
        bw_cell = numpy.pad(bw_cell, [(0, 0), (0, -extra_width)])

    bw_cell = cv2.resize(bw_cell, (CHAR_HEIGHT // 2, CHAR_HEIGHT // 2))
    bw_cell = utils.to_black_and_white(bw_cell)

    bw_cell = numpy.pad(bw_cell, [(1, 1), (1, 1)])

    return (features,
            tuple(utils.get_color(backgrounds)),
            tuple(utils.get_color(foregrounds)),
            (bw_cell / 255 if for_model else bw_cell))

def get_prediction(vector):
    label = numpy.argsort(vector)[-1]
    return chr(LABEL_TO_CHARACTER[label])

def predict(cells):
    initialize_mappings()
    initialize_character_models()
    # Run main model
    cells = [(numpy.array(list(map(lambda i: i[1][3], cells))))]
    result = character_models['main'].predict(cells)
    return list(map(lambda vector: get_prediction(vector), result))

def initialize_character_models():
    global initialized
    if initialized['models']:
        return
    character_models['main'] = models.load_model('model/character/model.keras')
    initialize_character_fallback_models('model/character/fallback', numpy.array([], dtype='int'))
    initialized['models'] = True

def initialize_character_fallback_models(base_path, stack):
    if not os.path.exists(base_path):
        return
    character_models[stack.tobytes()] = {}
    if len(stack) > 0:
        with open(base_path + '/labels.json', 'r') as file:
            data = json.load(file)
            character_models[stack.tobytes()]['data'] = data
        character_models[stack.tobytes()]['model'] = models.load_model(base_path + '/model.keras')
    for i in itertools.count():
        curr_path = base_path + '/' + str(i)
        if not os.path.exists(curr_path):
            return
        cpy = stack.copy()
        cpy = numpy.append(cpy, i)
        initialize_character_fallback_models(base_path + '/' + str(i), cpy)