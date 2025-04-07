from constants import CHAR_HEIGHT
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import cv2, numpy, os, time, utils

DATASET_CHARACTERS = []
BACKGROUNDS = []
FOREGROUNDS = []
CHARS_PER_LINE = 30
SPACING = 5

hashes = set()

def get_hash(arr):
    result = ''
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result = result + str(int(arr[i][j] / 255))
    return hash(result)

def create_missing_subdirectories(ordinal, font_str):
    # Create subdirectories if not exists
    if not os.path.exists('./dataset'):
        os.mkdir('dataset')
    if not os.path.exists('dataset/character'):
        os.mkdir('dataset/character')
    if not os.path.exists('./dataset/character/' + str(ordinal)):
        os.mkdir('dataset/character/' + str(ordinal))
    if not os.path.exists('./dataset/character/' + str(ordinal) + '/' + font_str):
        os.mkdir('dataset/character/' + str(ordinal) + '/' + font_str)

def initialize():
    for i in range(33, 127):
        DATASET_CHARACTERS.append(i)
    for gray in (25, 50, 75, 105):
        FOREGROUNDS.append((gray, gray, gray))
        BACKGROUNDS.append((255 - gray, 255 - gray, 255 - gray))

def generate(font_ref, font_str, char_height, background, foreground, quality):
    char_area = char_height * char_height // 2
    total_area = char_area * len(DATASET_CHARACTERS)
    width = (char_height * CHARS_PER_LINE) // 2
    height = (total_area // width) + char_height
    image = Image.new('RGB', (width, height), background)
    draw = ImageDraw.Draw(image)
    cnt_written = 0
    x, y = 0, 0
    for i in DATASET_CHARACTERS:
        draw.text((x, y), chr(i), fill = foreground, font = font_ref)
        cnt_written = cnt_written + 1
        if cnt_written % CHARS_PER_LINE == 0:
            x = 0
            y = y + char_height
        else:
            x = x + (char_height // 2)
    image.save('kek.jpg', quality = quality)
    split(char_height, font_str)

def split(char_height, font_str):
    image = cv2.imread('kek.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
    char_width = char_height // 2
    height = image.shape[0]
    width = image.shape[1]
    current = 0
    for i in range(0, height, char_height):
        for j in range(0, width, char_width):
            if current == len(DATASET_CHARACTERS):
                return
            symbol = image[i:(i + char_height), j:(j + char_width)]
            symbol = cv2.resize(symbol, (CHAR_HEIGHT // 2, CHAR_HEIGHT))
            _, symbol = cv2.threshold(symbol, 127, 255, cv2.THRESH_OTSU)
            nonzero = numpy.nonzero(symbol - 255)
            if len(nonzero[0]) == 0:
                current = current + 1
                continue
            row_l, row_r = numpy.min(nonzero[0]), numpy.max(nonzero[0]) + 1
            col_l, col_r = numpy.min(nonzero[1]), numpy.max(nonzero[1]) + 1
            symbol = symbol[row_l:row_r, col_l:col_r]
            output = cv2.resize(symbol, (CHAR_HEIGHT // 2, CHAR_HEIGHT))
            _, output = cv2.threshold(output, 127, 255, cv2.THRESH_OTSU)
            ordinal = DATASET_CHARACTERS[current]
            hsh = get_hash(output)
            if hsh in hashes:
                current = current + 1
                continue
            hashes.add(hsh)
            create_missing_subdirectories(ordinal, font_str)
            index = utils.file_cnt('dataset/character/' + str(ordinal) + '/' + font_str)
            cv2.imwrite('dataset/character/' + str(ordinal) + '/' + font_str + '/' + str(index) + '.jpg', output)
            current = current + 1

initialize()

def generate_all():
    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/')
        for char_size in (7, 8, 9, 10, 11, 12):
            start = time.time()
            print(font, char_size)
            font_ref = ImageFont.truetype(font_str, char_size * 2)
            for background in BACKGROUNDS:
                for foreground in FOREGROUNDS:
                    for quality in (20, 50, 80):
                        generate(font_ref, font_str.replace('fonts/', ''), (char_size + SPACING) * 2, background, foreground, quality)
            end = time.time()
            print('Taken', end - start, 's.')
generate_all()