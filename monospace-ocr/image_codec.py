import os.path

import numpy
from dahuffman import HuffmanCodec
from PIL import Image, ImageDraw, ImageFont

class Region:
    def __init__(
            self,
            font, # Font string, e.g. 'Consolas' or 'Consolas-Bold'
            coordinates, # Tuple of x- and y- coordinates of top-left corner
            split_size, # Tuple of horizontal and vertical sizes of splits
            splits, # Tuple of horizontal and vertical splits
            symbols, # 2D array of symbols
            foregrounds, # 2D array of foreground colors
            backgrounds, # 2D array of background colors
    ):
        self.font = font
        self.coordinates = coordinates
        self.split_size = split_size
        self.splits = splits
        self.symbols = symbols
        self.foregrounds = foregrounds
        self.backgrounds = backgrounds

def arr_8_to_int(arr):
    """
    Converts array of 8 binary values to its corresponding numeric value.
    E.g. [0, 1, 1, 1, 0, 0, 0, 0] converts to 14.
    :param arr: the input array
    :return: the corresponding numeric value.
    """
    return sum(int(x) << i for i, x in enumerate(arr))

def arr_to_ints(arr):
    """
    Converts array of k binary values to array of numeric values of sub-arrays of size 8.
    :param arr: the input array
    :return: the corresponding numeric value
    """
    result = []
    for i in range(0, len(arr), 8):
        result.append(arr_8_to_int(arr[i:(i + 8)]))
    return result

def combine_bytes(arr):
    result = 0
    for x in arr:
        result = result * 256
        result = result + x
    return result

def find_font(font):
    """
    Finds the relative path of the input font (or, possibly, one of its variants).
    :param font: the font string
    :return: the relative path of the found font, or None if none was found
    """
    for suffix in ['', '-Regular', '-Bold', '-Italic']:
        for ending in ['ttf', 'otf']:
            if os.path.exists('fonts/' + font + suffix + '.' + ending):
                return 'fonts/' + font + suffix + '.' + ending
    return None

def is_similar_color(a, b):
    for i in range(3):
        if max(a[i], b[i]) - min(a[i], b[i]) > 10:
            return False
    return True

def generate_region(region, image):
    """
    Draws given region on given image.
    :param region: the input region
    :param image: the input image
    :return: the image with the input region drawn.
    """
    draw = ImageDraw.Draw(image)
    index = 0
    row_splits = region.splits[1]
    col_splits = region.splits[0]
    font_str = find_font(region.font)
    font_ref = ImageFont.truetype(font_str, region.split_size[0] * 2 - 2)
    width = region.split_size[0]
    height = region.split_size[1]
    len_w = len(row_splits) - 1
    len_h = len(col_splits) - 1
    color_mapping = {}
    for j in range(len_w):
        for i in range(len_h):
            found = False
            for col in color_mapping:
                if is_similar_color(col, tuple(region.backgrounds[index])):
                    color_mapping[tuple(region.backgrounds[index])] = col
                    found = True
                    break
            if not found:
                color_mapping[tuple(region.backgrounds[index])] = tuple(region.backgrounds[index])
            found = False
            for col in color_mapping:
                if is_similar_color(col, tuple(region.foregrounds[index])):
                    color_mapping[tuple(region.foregrounds[index])] = col
                    found = True
                    break
            if not found:
                color_mapping[tuple(region.foregrounds[index])] = tuple(region.foregrounds[index])
            if region.symbols[index] == 'Y':
                region.symbols[index] = 'y'
            draw.rectangle(xy = (i * width, j * height, (i + 1) * width, (j + 1) * height), fill = color_mapping[tuple(region.backgrounds[index])])
            draw.text(xy = (i * width, j * height), text = region.symbols[index], fill = color_mapping[tuple(region.foregrounds[index])], font = font_ref)

            index = index + 1

    return image

def encode(regions):
    symbol_frequencies = {}
    color_frequencies = {}
    color_conversion = {}

    for region in regions:
        for symbol in region.symbols:
            symbol_frequencies[symbol] = symbol_frequencies.get(symbol, 0) + 1
        for color in (numpy.concat((region.backgrounds, region.foregrounds))):
            color = (color[0], color[1], color[2])
            found = False
            for col in color_frequencies:
                if is_similar_color(color, col):
                    color_frequencies[col] = color_frequencies[col] + 1
                    color_conversion[color] = col
                    found = True
            if not found:
                color_frequencies[color] = 1

    symbol_huffman = HuffmanCodec.from_frequencies(symbol_frequencies)
    color_huffman = HuffmanCodec.from_frequencies(color_frequencies)

    data = numpy.array([], dtype = 'uint8')

    # Frequency data
    data = numpy.append(data, list(len(symbol_frequencies).to_bytes(1)))
    for symbol in symbol_frequencies:
        data = numpy.append(data, list(ord(symbol).to_bytes(1)))
        data = numpy.append(data, list(symbol_frequencies[symbol].to_bytes(3)))
    data = numpy.append(data, list(len(color_frequencies).to_bytes(1)))
    for color in color_frequencies:
        data = numpy.append(data, list(int(color[0]).to_bytes(1)))
        data = numpy.append(data, list(int(color[1]).to_bytes(1)))
        data = numpy.append(data, list(int(color[2]).to_bytes(1)))
        data = numpy.append(data, list(color_frequencies[color].to_bytes(3)))

    data = numpy.append(data, list(len(regions).to_bytes(1)))

    for region in regions:
        # Font str
        font_bytes = region.font.encode('utf-8')
        data = numpy.append(data, list(len(font_bytes).to_bytes(1)))
        data = numpy.append(data, list(font_bytes))
        # Coordinates
        data = numpy.append(data, list(region.coordinates[0].to_bytes(2)))
        data = numpy.append(data, list(region.coordinates[1].to_bytes(2)))
        for axis in range(2):
            # Split sizes
            data = numpy.append(data, list(int(region.split_size[axis]).to_bytes(2)))
            data = numpy.append(data, list(int(region.splits[axis][0]).to_bytes(2)))
            # Split coordinates (delta compared to the expected step size)
            splits = numpy.array([])
            for i in range(1, len(region.splits[axis])):
                delta = (region.splits[axis][i] - region.splits[axis][i - 1]) - region.split_size[axis]
                if delta == 0:
                    splits = numpy.append(splits, [0])
                else:
                    splits = numpy.append(splits, [1])
            splits_bytes = arr_to_ints(splits)
            data = numpy.append(data, list(len(splits_bytes).to_bytes(2)))
            data = numpy.append(data, splits_bytes)
        # Symbols
        symbols_bytes = list(symbol_huffman.encode(region.symbols))
        data = numpy.append(data, symbols_bytes)
        # Colors
        res = []
        for col in numpy.concat((region.backgrounds, region.foregrounds)):
            col = tuple(col)
            if col in color_frequencies:
                res.append(col)
            else:
                res.append(color_conversion[col])
        color_data = color_huffman.encode(res)
        data = numpy.append(data, list(color_data))

    return bytes(list(data))

def decode(data, output_file):
    symbol_frequencies = {}
    color_frequencies = {}

    # Frequency data
    index = 1
    symbols_cnt = data[0]
    # Symbol frequencies
    for i in range(symbols_cnt):
        symbol = data[index]
        frequency = combine_bytes(data[(index + 1):(index + 4)])
        symbol_frequencies[symbol] = frequency
        index = index + 4
    colors_cnt = data[index]
    index = index + 1
    # Color frequencies
    for i in range(colors_cnt):
        color = (data[index], data[index + 1], data[index + 2])
        frequency = combine_bytes(data[(index + 3):(index + 6)])
        color_frequencies[color] = frequency
        index = index + 6

    symbol_huffman = HuffmanCodec.from_frequencies(symbol_frequencies)
    color_huffman = HuffmanCodec.from_frequencies(color_frequencies)

    regions_cnt = data[index]
    index = index + 1

    for i in range(regions_cnt):
        length = data[index]
        font_bytes = data[(index + 1):(index + length + 1)]
        font_str = font_bytes.decode('utf-8')

        index = index + length + 1
        coordinate_x = combine_bytes(data[index:(index + 2)])
        coordinate_y = combine_bytes(data[(index + 2):(index + 4)])
        print(coordinate_x, coordinate_y)


