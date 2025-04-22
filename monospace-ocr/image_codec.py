import numpy
from dahuffman import HuffmanCodec
from PIL import Image, ImageDraw, ImageFont

class Region:
    def __init__(self, coordinates, split_size, splits, symbols, foregrounds, backgrounds):
        self.coordinates = coordinates
        self.split_size = split_size
        self.splits = splits
        self.symbols = symbols
        self.foregrounds = foregrounds
        self.backgrounds = backgrounds

def arr_8_to_int(arr):
    return sum(int(x) << i for i, x in enumerate(arr))

def arr_to_ints(arr):
    result = []
    for i in range(0, len(arr), 8):
        result.append(arr_8_to_int(arr[i:(i + 8)]))
    return result

def is_similar_color(a, b):
    for i in range(3):
        if max(a[i], b[i]) - min(a[i], b[i]) > 15:
            return False
    return True

def generate_region(region, image):
    draw = ImageDraw.Draw(image)
    index = 0
    row_splits = region.splits[1]
    col_splits = region.splits[0]
    font_ref = ImageFont.truetype('fonts/Consolas-Regular.ttf', region.split_size[0] * 2 - 2)
    kek_ref = ImageFont.truetype('fonts/Consolas-Italic.ttf', region.split_size[0] * 2 - 2)
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
            if 15 <= i <= 17 and j == 7:
                draw.text(xy = (i * width, j * height), text = region.symbols[index], fill = color_mapping[tuple(region.foregrounds[index])], font = kek_ref)
            else:
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
    for color in color_frequencies:
        data = numpy.append(data, list(int(color[0]).to_bytes(1)))
        data = numpy.append(data, list(int(color[1]).to_bytes(1)))
        data = numpy.append(data, list(int(color[2]).to_bytes(1)))
        data = numpy.append(data, list(color_frequencies[color].to_bytes(3)))

    for region in regions:
        for axis in range(2):
            # Split sizes
            data = numpy.append(data, list(int(region.split_size[axis]).to_bytes(2)))
            data = numpy.append(data, list(int(region.splits[axis][0]).to_bytes(2)))
            # Split coordinates (delta compared to the expected step size)
            splits = numpy.array([])
            for i in range(1, len(region.splits[axis])):
                delta = (region.splits[axis][i] - region.splits[axis][i - 1]) - region.split_size[axis]
                if delta == -1:
                    splits = numpy.append(splits, [1, 0])
                elif delta == 0:
                    splits = numpy.append(splits, [0])
                else:
                    splits = numpy.append(splits, [1, 1])
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