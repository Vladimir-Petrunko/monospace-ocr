import itertools

import utils
import numpy
from pathlib import Path
from tensorflow.keras import models
from dto import CellDto

CHARACTER_TO_LABEL = {}
LABEL_TO_CHARACTER = {}

initialized = {'mappings': False, 'models': False}
character_model = {}

def initialize_mappings():
    global initialized
    if initialized['mappings']:
        return
    for character in range(33, 127):
        index = len(CHARACTER_TO_LABEL)
        CHARACTER_TO_LABEL[character] = index
        LABEL_TO_CHARACTER[index] = character
    initialized['mappings'] = True

def get_prediction(vector, features):
    label = numpy.argsort(vector)[-1]
    candidate = chr(LABEL_TO_CHARACTER[label])
    return candidate

def predict(data):
    initialize()
    cells = list(map(lambda cell: cell[1].cell, data))
    features = list(map(lambda cell: cell[1].features, data))
    result = character_model['main'].predict(numpy.array(cells))

    cells_cnt = len(result)
    answer = [' ' for _ in range(cells_cnt)]
    for i in range(cells_cnt):
        answer[i] = get_prediction(result[i], features[i])

    return answer

def initialize_character_models():
    global initialized, character_model
    if initialized['models']:
        return
    character_model['main'] = models.load_model('model/character/model.keras')
    initialized['models'] = True

def post_analysis(result, height, width):
    """
    Performs lexical post-analysis on predicted symbols
    :param result: the symbol array
    :param height: the number of rows
    :param width: the number of columns
    :return: the corrected symbols, after post-analysis
    """
    for i in range(height):
        for j in range(width):
            result[i * height + j] = result[i * height + j]
    return result

def initialize():
    initialize_mappings()
    initialize_character_models()