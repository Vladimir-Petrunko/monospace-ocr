from pathlib import Path
from tensorflow.keras import models

CHARACTER_MODEL = models.load_model('model/character.keras')

FONT_TO_LABEL = {}
LABEL_TO_FONT = {}

CHARACTER_TO_LABEL = {}
LABEL_TO_CHARACTER = {}

initialized = False

def initialize_mappings():
    global initialized
    if initialized:
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
    initialized = True
