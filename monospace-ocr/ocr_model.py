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

def connected(cell):
    for i in range(cell.shape[0]):
        if cell[i][0] == cell[i][1] == 1:
            return True
    return False

def cut(cell):
    cell = cell / 255
    cell = 1 - cell

    if numpy.sum(cell[:, :1]) > numpy.sum(cell[:, 1:2]) and not connected(cell[:, :2]):
        cell = cell[:, 1:]
    if numpy.sum(cell[:, -1:]) > numpy.sum(cell[:, -2:-1]) and not connected(cell[:, -2:]):
        cell = cell[:, :-1]

    cell = 1 - cell
    cell = cell * 255
    return cell

def flood_adjust(cell):
    for i in range(cell.shape[0]):
        for j in range(cell.shape[1]):
            if cell[i][j] == 0:
                mask = flood(cell, (i, j))
                if len(numpy.nonzero(mask)[0]) <= 2:
                    cell[mask == 1] = 255
    return cell

def normalize(cell, is_grayscale, for_model = False):
    if not is_grayscale:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    black_and_white = utils.to_black_and_white(cell)
    if utils.get_background_black_and_white(black_and_white) == 0:
        cell = 255 - cell

    cell = 255 - cell
    flat = cell.flatten()
    min_color, max_color = numpy.min(flat), numpy.max(flat)
    if min_color == max_color:
        return None
    cell = cell - min_color
    cell = cell * (255 / (max_color - min_color)) * 1.25
    cell[cell > 255] = 255
    cell = 255 - cell

    cell = cell.astype('uint8')
    cell = utils.to_black_and_white(cell)

    mask = numpy.zeros(cell.shape) + 1
    for i in range(2, cell.shape[0] - 2):
        for j in range(2, cell.shape[1] - 2):
            if cell[i][j] == 0 and mask[i][j] == 1:
                res = flood(cell, (i, j))
                mask[res == 1] = 0

    mask[2:(cell.shape[0] - 2), 2:(cell.shape[1] - 2)] = 0
    cell[mask == 1] = 255

    if utils.is_monochrome(cell):
        return None

    nonzero = numpy.nonzero(cell - 255)
    row_l, row_r = numpy.min(nonzero[0]), numpy.max(nonzero[0])
    col_l, col_r = numpy.min(nonzero[1]), numpy.max(nonzero[1])
    cell = cell[row_l:(row_r + 1), col_l:(col_r + 1)]
    sz = max(row_r - row_l + 1, col_r - col_l + 1)
    cell = numpy.pad(cell, [(0, sz - (row_r - row_l + 1)), (0, sz - (col_r - col_l + 1))], mode = 'constant', constant_values = 255)

    cell = cv2.resize(cell, (CHAR_HEIGHT // 2, CHAR_HEIGHT // 2))
    cell = utils.to_black_and_white(cell)

    cell = numpy.pad(cell, [(1, 1), (1, 1)], mode = 'constant', constant_values = 255)

    if for_model:
        cell = 1 - (cell / 255)

    return cell

def get_prediction(vector):
    label = numpy.argsort(vector)[-1]
    return chr(LABEL_TO_CHARACTER[label])

def predict(cells):
    initialize_mappings()
    initialize_character_models()
    # Run main model
    cells = [(numpy.array(list(map(lambda i: i[1], cells))))]
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