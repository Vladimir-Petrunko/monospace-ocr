from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import cv2, numpy, os, time, utils, ocr_model

DATASET_CHARACTERS = []
BACKGROUNDS = []
FOREGROUNDS = []
CHARS_PER_LINE = 30
SPACING = 15

hashes = {}
indices = {}

feature_list = []

def get_hash(arr):
    return hash(arr.tobytes())

def create_missing_subdirectories():
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    if not os.path.exists('dataset/character'):
        os.mkdir('dataset/character')
    if not os.path.exists('dataset/character/full'):
        os.mkdir('dataset/character/full')

    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/').replace('fonts/', '')
        for i in DATASET_CHARACTERS:
            for s in ['full']:
                if not os.path.exists('dataset/character/' + s + '/' + str(i)):
                    os.mkdir('dataset/character/' + s + '/' + str(i))
                if not os.path.exists('dataset/character/' + s + '/' + str(i) + '/' + font_str):
                    os.mkdir('dataset/character/' + s + '/' + str(i) + '/' + font_str)
                index = utils.file_cnt('dataset/character/' + s + '/' + str(i) + '/' + font_str)
                indices[s + '/' + str(i) + '/' + font_str] = index

def initialize():
    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/').replace('fonts/', '')
        trimmed = font_str[:font_str.index('-')]
        print(trimmed)
        hashes[trimmed] = []
        for i in range(33, 127):
            hashes[trimmed].append(set())
    for i in range(33, 127):
        DATASET_CHARACTERS.append(i)
    for gray in [0, 30]:
        FOREGROUNDS.append((gray, gray, gray))
        BACKGROUNDS.append((255 - gray, 255 - gray, 255 - gray))

def generate(font_ref, font_str, char_size, char_height, background, foreground, quality):
    char_area = char_height * char_height // 2
    total_area = char_area * len(DATASET_CHARACTERS)
    width = (char_height * CHARS_PER_LINE) // 2
    height = (total_area // width) + char_height
    image = Image.new('RGB', (width, height), background)
    draw = ImageDraw.Draw(image)
    cnt_written = 0
    x, y = 0, 0
    for i in DATASET_CHARACTERS:
        draw.text((x + 1, y + 1), chr(i), fill = foreground, font = font_ref)
        cnt_written = cnt_written + 1
        if cnt_written % CHARS_PER_LINE == 0:
            x = 0
            y = y + char_height
        else:
            x = x + (char_height // 2)
    image.save('kek.jpg', quality = quality)
    split(char_size, char_height, font_str)

def split(char_size, char_height, font_str):
    trimmed = font_str[:font_str.index('-')]
    image = cv2.imread('kek.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    char_width = char_height // 2
    height = image.shape[0]
    width = image.shape[1]
    current = 0
    for i in range(0, height, char_height):
        for j in range(0, width, char_width):
            if current == len(DATASET_CHARACTERS):
                return
            ordinal = DATASET_CHARACTERS[current]
            symbol = image[i:(i + char_size * 2 + 2), j:(j + char_size + 2)]
            features, _, _, symbol = utils.normalize(symbol, is_grayscale = True, for_model = False)
            if symbol is not None:
                hsh = get_hash(symbol)
                features = numpy.append(features, ordinal)
                feature_list.append(features)
                if hsh not in hashes[trimmed][ordinal - 33]:
                    hashes[trimmed][ordinal - 33].add(hsh)
                    index = indices['full/' + str(ordinal) + '/' + font_str]
                    cv2.imwrite('dataset/character/full/' + str(ordinal) + '/' + font_str + '/' + str(index) + '.jpg', symbol)
                    indices['full/' + str(ordinal) + '/' + font_str] = index + 1
            current = current + 1

def generate_all():
    fonts = Path('fonts')
    for font in fonts.iterdir():
        font_str = str(font).replace('\\', '/')
        for char_size in (6, 7, 8, 9, 10, 12):
            start = time.time()
            print(font, char_size)
            font_ref = ImageFont.truetype(font_str, char_size * 2)
            for background in BACKGROUNDS:
                for foreground in FOREGROUNDS:
                    for quality in [90]:
                        generate(font_ref, font_str.replace('fonts/', ''), char_size, (char_size + SPACING) * 2, background, foreground, quality)
            end = time.time()
            print('Time taken:', end - start, 's')

initialize()
create_missing_subdirectories()
generate_all()

feature_list = numpy.array(feature_list)
numpy.savetxt("features.csv", feature_list, delimiter = ',')