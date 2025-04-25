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

MAX_SIZE = 15

FONT_TO_LABEL = {}
LABEL_TO_FONT = {}

CHARACTER_TO_LABEL = {}
LABEL_TO_CHARACTER = {}

initialized = {'mappings': False, 'models': False}

character_models = {}
fallback_backward_mappings = {}

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

def get_character_connected_components(character_errors):
    markers = []
    components = []
    sorted_indices = numpy.argsort([len(component) for component in character_errors])
    for i in sorted_indices:
        new_component = []
        for ch in character_errors[i]:
            new_component.append(ch)
        if len(new_component) == 0:
            continue
        if len(components) == 0:
            components.append(set(new_component))
            components[-1].add(int(i))
            markers.append([int(i)])
        else:
            merged_component = components[-1].copy()
            for it in new_component:
                merged_component.add(it)
            if len(merged_component) <= MAX_SIZE:
                components[-1] = merged_component
                markers[-1].append(int(i))
            else:
                components.append(set(new_component))
                markers.append([int(i)])
        components[-1].add(int(i))
    return markers, components

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
    bw_cell = bw_cell * (255 / (max_color - min_color) * 1.25)
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
    cells = numpy.array(list(map(lambda i: i[1][3], cells)))

    result = character_models['main'].predict(cells)
    answer = list(map(lambda vector: get_prediction(vector), result))

    for i in range(len(answer)):
        ch = ord(answer[i])
        #if (ch - 33) in character_models:
            #obj = character_models[ch - 33]
            #vec = obj[1].predict(cells[i:(i + 1)])[0]
            #label = numpy.argsort(vec)[-1]
            #answer[i] = chr(fallback_backward_mappings[obj[0]][label])

    return answer

def initialize_character_models():
    global initialized
    if initialized['models']:
        return
    character_models['main'] = models.load_model('model/character/model.keras')
    initialize_character_fallback_models()
    initialized['models'] = True

def initialize_character_fallback_models():
    for i in itertools.count():
        if os.path.exists('model/character/fallback/' + str(i)):
            fallback_model = models.load_model('model/character/fallback/' + str(i) + '/model.keras')
            backward_mappings = {}
            with open('model/character/fallback/' + str(i) + '/markers.json', 'r') as file:
                data = json.load(file)
                markers = data['markers']
                for marker in markers:
                    character_models[marker] = [i, fallback_model]
            with open('model/character/fallback/' + str(i) + '/labels.json', 'r') as file:
                data = json.load(file)
                for key in data:
                    value = data[key]
                    backward_mappings[value] = int(key)
            fallback_backward_mappings[i] = backward_mappings
        else:
            break
